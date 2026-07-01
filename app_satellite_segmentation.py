"""Gradio viewer for the custom satellite semantic segmentation model."""

from __future__ import annotations

from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from src.dino_semantic_distillation import DinoSemanticDistillationModel
from src.utils import load_config
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50


DEFAULT_CHECKPOINT = "checkpoints/satellite_segmentation_v13/best.pth"
DEFAULT_DINO_CONFIG = "configs/dino_semantic_distill_v12.yaml"
DEFAULT_DINO_CHECKPOINT = "checkpoints/dino_semantic_distill_v12/best.pth"
DEFAULT_PORT = 7862


PALETTE = np.array(
    [
        [255, 255, 255],  # road
        [0, 90, 255],     # building
        [0, 220, 220],    # grass
        [20, 170, 40],    # tree
        [255, 230, 0],    # car
        [220, 40, 40],    # other
    ],
    dtype=np.uint8,
)


class SegmentationModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return out["out"]
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_names = ["road", "building", "grass", "tree", "car", "other"]
loaded_checkpoint = None
dino_model = None
loaded_dino_checkpoint = None
dino_class_names = ["road", "building", "grass", "tree", "car", "other", "sky"]


def build_model(num_classes: int) -> nn.Module:
    model_inner = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes, aux_loss=False)
    return SegmentationModelWrapper(model_inner)


def load_model(checkpoint_path: str):
    global model, class_names, loaded_checkpoint
    checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
    checkpoint_path = str(Path(checkpoint_path))
    if model is not None and loaded_checkpoint == checkpoint_path:
        return model

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = ckpt.get("classes", class_names)
    palette = ckpt.get("palette")
    if palette is not None:
        global PALETTE
        PALETTE = np.array(palette, dtype=np.uint8)
    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    loaded_checkpoint = checkpoint_path
    print(f"[SatelliteSeg UI] Loaded {checkpoint_path} on {device}", flush=True)
    return model


def load_dino_model(config_path: str, checkpoint_path: str):
    global dino_model, loaded_dino_checkpoint, dino_class_names
    config_path = config_path or DEFAULT_DINO_CONFIG
    checkpoint_path = checkpoint_path or DEFAULT_DINO_CHECKPOINT
    if dino_model is not None and loaded_dino_checkpoint == checkpoint_path:
        return dino_model

    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    num_classes = int(model_cfg.get("num_classes", 7))
    dino_model = DinoSemanticDistillationModel(
        num_classes=num_classes,
        dino_model_name=model_cfg.get("dino_model_name", "vit_small_patch16_224.dino"),
        dino_pretrained=bool(model_cfg.get("dino_pretrained", True)),
        feature_channels=int(model_cfg.get("feature_channels", 128)),
        head_hidden_channels=int(model_cfg.get("head_hidden_channels", 128)),
        share_overhead_backbone=bool(model_cfg.get("share_overhead_backbone", True)),
        freeze_dino=bool(model_cfg.get("freeze_dino", True)),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dino_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    dino_model.eval()
    loaded_dino_checkpoint = checkpoint_path
    dino_class_names = ckpt.get("classes", dino_class_names[:num_classes])
    if len(dino_class_names) != num_classes:
        dino_class_names = [f"class_{i}" for i in range(num_classes)]
    print(f"[SatelliteSeg UI] Loaded DINO semantic model {checkpoint_path} on {device}", flush=True)
    return dino_model


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]


def preprocess(image: np.ndarray, image_size: int):
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    original = image.astype(np.uint8)
    pil = Image.fromarray(original).convert("RGB")
    resized = TF.resize(pil, [image_size, image_size], interpolation=TF.InterpolationMode.BILINEAR)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return original, tensor.unsqueeze(0)


def replace_other_by_threshold(logits: torch.Tensor, names, replace_other: bool, threshold: float) -> torch.Tensor:
    pred = logits.argmax(dim=0)
    invalid_names = {"other", "sky"}
    invalid_indices = [idx for idx, name in enumerate(names) if name in invalid_names]
    if invalid_indices and len(invalid_indices) < len(names):
        non_other_logits = logits.clone()
        for invalid_idx in invalid_indices:
            non_other_logits[invalid_idx] = -1e9
        second_choice = non_other_logits.argmax(dim=0)
        # Satellite outputs should not keep invalid classes. For DeepLabV3 this
        # removes "other"; for DINO it removes both "other" and satellite-impossible
        # "sky". The UI threshold is kept only for backward compatibility.
        replace_mask = torch.zeros_like(pred, dtype=torch.bool)
        for invalid_idx in invalid_indices:
            replace_mask |= pred == invalid_idx
        pred = torch.where(replace_mask, second_choice, pred)
    return pred


@torch.no_grad()
def segment_satellite(image, checkpoint_path, image_size, overlay_alpha, replace_other, other_threshold):
    if image is None:
        return None, None, None, "请先上传卫星图。"

    try:
        image_size = int(image_size)
        net = load_model(checkpoint_path)
        original, tensor = preprocess(np.array(image), image_size)
        tensor = tensor.to(device)
        logits = net(tensor)
        pred_logits = logits[0]
        pred_small_t = replace_other_by_threshold(pred_logits, class_names, replace_other, other_threshold)

        pred_small = pred_small_t.cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred_small, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_rgb = colorize_mask(pred)
        overlay = np.clip((1 - overlay_alpha) * original + overlay_alpha * mask_rgb, 0, 255).astype(np.uint8)

        total = pred.size
        lines = [
            f"设备 / Device: {device}",
            f"输入尺寸 / Input size: {image_size}",
            f"other 替换 / Replace other: {bool(replace_other)}",
            f"other 保留阈值 / Other keep threshold: {float(other_threshold):.2f}",
        ]
        for idx, name in enumerate(class_names):
            ratio = float((pred == idx).sum()) / max(1, total)
            lines.append(f"{name}: {ratio * 100:.2f}%")
        status = "\n".join(lines)
        return mask_rgb, overlay, pred, status
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return None, None, None, f"推理失败 / Inference failed: {exc}"


@torch.no_grad()
def compare_dino_deeplab(
    image,
    deeplab_checkpoint,
    dino_config,
    dino_checkpoint,
    image_size,
    overlay_alpha,
    replace_other,
    other_threshold,
):
    if image is None:
        return None, None, None, None, None, "请先上传卫星图。"
    try:
        image_size = int(image_size)
        original, tensor = preprocess(np.array(image), image_size)
        tensor = tensor.to(device)

        deeplab = load_model(deeplab_checkpoint)
        deeplab_logits = deeplab(tensor)[0]
        deeplab_pred_small = replace_other_by_threshold(deeplab_logits, class_names, replace_other, other_threshold)

        dino = load_dino_model(dino_config, dino_checkpoint)
        dino_logits = dino(satellite_rgb=tensor)["satellite_logits"][0]
        dino_pred_small = replace_other_by_threshold(dino_logits, dino_class_names, replace_other, other_threshold)

        h, w = original.shape[:2]
        deeplab_pred = cv2.resize(deeplab_pred_small.cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        dino_pred = cv2.resize(dino_pred_small.cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        deeplab_mask = colorize_mask(deeplab_pred)
        dino_mask = colorize_mask(dino_pred)
        deeplab_overlay = np.clip((1 - overlay_alpha) * original + overlay_alpha * deeplab_mask, 0, 255).astype(np.uint8)
        dino_overlay = np.clip((1 - overlay_alpha) * original + overlay_alpha * dino_mask, 0, 255).astype(np.uint8)
        comparison = np.concatenate([original, dino_mask, deeplab_mask, dino_overlay, deeplab_overlay], axis=1)

        def ratios(mask, names):
            total = mask.size
            return [f"{name}: {(mask == idx).sum() / max(1, total) * 100:.2f}%" for idx, name in enumerate(names)]

        status = "\n".join(
            [
                f"设备 / Device: {device}",
                f"other 保留阈值 / Other keep threshold: {float(other_threshold):.2f}",
                "",
                "[DINO v12]",
                *ratios(dino_pred, dino_class_names),
                "",
                "[DeepLabV3]",
                *ratios(deeplab_pred, class_names),
            ]
        )
        return dino_mask, deeplab_mask, dino_overlay, deeplab_overlay, comparison, status
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return None, None, None, None, None, f"对比失败 / Compare failed: {exc}"


def create_interface():
    with gr.Blocks(title="Satellite Semantic Segmentation") as demo:
        gr.Markdown(
            """
            # 卫星图语义分割测试 / Satellite Semantic Segmentation

            模型：DINO v12 semantic distillation vs DeepLabV3-ResNet50  
            类别：road, building, grass, tree, car, other

            上传一张卫星图，同时查看 DINO 语义分割和 DeepLabV3 语义分割的效果。
            """
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="卫星图 / Satellite image", type="numpy", height=360)
                checkpoint_input = gr.Textbox(label="DeepLabV3 权重 / DeepLabV3 checkpoint", value=DEFAULT_CHECKPOINT)
                dino_config_input = gr.Textbox(label="DINO 配置 / DINO config", value=DEFAULT_DINO_CONFIG)
                dino_checkpoint_input = gr.Textbox(label="DINO 权重 / DINO checkpoint", value=DEFAULT_DINO_CHECKPOINT)
                image_size_input = gr.Slider(label="推理尺寸 / Inference size", minimum=128, maximum=512, value=256, step=32)
                overlay_alpha_input = gr.Slider(label="叠加透明度 / Overlay alpha", minimum=0.0, maximum=1.0, value=0.45, step=0.05)
                replace_other_input = gr.Checkbox(
                    label="启用 other 阈值替换 / Enable other threshold replacement",
                    value=True,
                )
                other_threshold_input = gr.Slider(
                    label="other 去除阈值 / Other removal threshold",
                    minimum=0.0,
                    maximum=1.01,
                    value=1.01,
                    step=0.01,
                    info="当前设置为去除 other：预测为 other 的像素会替换成非 other 的最高概率类别。",
                )
                run_btn = gr.Button("开始对比 / Compare DINO and DeepLabV3", variant="primary")
            with gr.Column():
                dino_mask_output = gr.Image(label="DINO v12 语义分割 / DINO semantic mask", height=300)
                deeplab_mask_output = gr.Image(label="DeepLabV3 语义分割 / DeepLabV3 semantic mask", height=300)
                status_output = gr.Textbox(label="类别比例 / Class ratios", lines=14)
        with gr.Row():
            dino_overlay_output = gr.Image(label="DINO 叠加图 / DINO overlay", height=300)
            deeplab_overlay_output = gr.Image(label="DeepLabV3 叠加图 / DeepLabV3 overlay", height=300)
        comparison_output = gr.Image(
            label="对比图：原图 | DINO mask | DeepLabV3 mask | DINO overlay | DeepLabV3 overlay",
            height=280,
        )
        label_output = gr.Image(label="类别 ID 图 / Label ID map", visible=False)
        run_btn.click(
            fn=compare_dino_deeplab,
            inputs=[
                image_input,
                checkpoint_input,
                dino_config_input,
                dino_checkpoint_input,
                image_size_input,
                overlay_alpha_input,
                replace_other_input,
                other_threshold_input,
            ],
            outputs=[
                dino_mask_output,
                deeplab_mask_output,
                dino_overlay_output,
                deeplab_overlay_output,
                comparison_output,
                status_output,
            ],
        )
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=DEFAULT_PORT, share=False, show_error=True)
