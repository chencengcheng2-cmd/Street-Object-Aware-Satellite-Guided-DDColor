"""
Gradio web interface for Satellite-Guided DDColor.
"""

import gradio as gr
import numpy as np
import torch
import cv2
from pathlib import Path
from datetime import datetime
import copy
import subprocess
import sys
import yaml

from src.dataset import CVUSADataset
from src.model import SatelliteGuidedDDColor
from src.utils import load_config, load_matching_state_dict
from src.polar_transform import create_polar_from_satellite
from inference import colorize_panorama, prepare_panorama, default_patch_indices


# Global model
DEFAULT_CONFIG_PATH = "configs/satellite_constrained_v17.yaml"
DEFAULT_CHECKPOINT_PATH = "checkpoints\\satellite_constrained_v17_server\\best.pth"
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
config = None
training_process = None
loaded_checkpoint_path = None
loaded_use_lane_vit = None
loaded_use_satellite_vit = None
loaded_use_cross_view_vit = None
loaded_use_semantic_cross_view_vit = None
loaded_use_semantic_color_token_match = None
loaded_use_polar_token_match = None


def _build_model(model_config):
    return SatelliteGuidedDDColor(
        ddcolor_weights_path=model_config['ddcolor']['weights_path'],
        ddcolor_code_path=model_config.get('ddcolor', {}).get('code_path'),
        context_dim=model_config['model']['context_dim'],
        polar_encoder_pretrained=model_config['model']['polar_encoder_pretrained'],
        satellite_encoder_pretrained=model_config['model'].get('satellite_encoder_pretrained', True),
        correction_type=model_config['model']['correction_type'],
        residual_scale=model_config['model']['residual_scale'],
        use_polar_context=model_config['model'].get('use_polar_context', True),
        use_lane_vit=model_config['model'].get('use_lane_vit', False),
        lane_vit_embed_dim=model_config['model'].get('lane_vit_embed_dim', 192),
        lane_vit_depth=model_config['model'].get('lane_vit_depth', 4),
        lane_vit_heads=model_config['model'].get('lane_vit_heads', 3),
        lane_vit_patch_size=model_config['model'].get('lane_vit_patch_size', 16),
        lane_feature_dim=model_config['model'].get('lane_feature_dim', 64),
        use_satellite_vit=model_config['model'].get('use_satellite_vit', False),
        satellite_vit_embed_dim=model_config['model'].get('satellite_vit_embed_dim', 192),
        satellite_vit_depth=model_config['model'].get('satellite_vit_depth', 4),
        satellite_vit_heads=model_config['model'].get('satellite_vit_heads', 3),
        satellite_vit_patch_size=model_config['model'].get('satellite_vit_patch_size', 16),
        satellite_vit_feature_dim=model_config['model'].get('satellite_vit_feature_dim', 64),
        use_cross_view_vit=model_config['model'].get('use_cross_view_vit', False),
        cross_view_embed_dim=model_config['model'].get('cross_view_embed_dim', 192),
        cross_view_depth=model_config['model'].get('cross_view_depth', 3),
        cross_view_heads=model_config['model'].get('cross_view_heads', 3),
        use_semantic_cross_view_vit=model_config['model'].get('use_semantic_cross_view_vit', False),
        use_semantic_color_token_match=model_config['model'].get('use_semantic_color_token_match', False),
        use_polar_token_match=model_config['model'].get('use_polar_token_match', False),
        use_semantic_cnn_context=model_config['model'].get('use_semantic_cnn_context', False),
        semantic_num_classes=model_config['model'].get('semantic_num_classes', 6),
        cross_view_patch_size=model_config['model'].get('cross_view_patch_size', 16),
        cross_view_street_patch_size=model_config['model'].get('cross_view_street_patch_size', model_config['model'].get('cross_view_patch_size', 16)),
        cross_view_satellite_patch_size=model_config['model'].get('cross_view_satellite_patch_size', 8),
        cross_view_feature_dim=model_config['model'].get('cross_view_feature_dim', 64),
        color_token_match_weight=model_config['model'].get('color_token_match_weight', 3.0),
        semantic_distribution_weight=model_config['model'].get('semantic_distribution_weight', 2.0),
        boundary_match_weight=model_config['model'].get('boundary_match_weight', 2.0),
        token_delta_scale=model_config['model'].get('token_delta_scale', 0.35),
        token_correction_scale=model_config['model'].get('token_correction_scale', 0.8),
        street_object_hidden_dim=model_config['model'].get('street_object_hidden_dim', 96),
        street_object_num_masks=model_config['model'].get('street_object_num_masks', 8),
        street_object_detail_scale=model_config['model'].get('street_object_detail_scale', 0.18),
        satellite_prior_strength=model_config['model'].get('satellite_prior_strength', 0.65),
        use_street_gray_edges=model_config['model'].get('use_street_gray_edges', False),
        use_street_gray_modulation=model_config['model'].get('use_street_gray_modulation', True),
        use_gray_satellite_token_selection=model_config['model'].get('use_gray_satellite_token_selection', True),
        use_satellite_chroma_token_selection=model_config['model'].get('use_satellite_chroma_token_selection', False),
        token_selection_patch_size=model_config['model'].get('token_selection_patch_size', 16),
        token_selection_dim=model_config['model'].get('token_selection_dim', 32),
        gray_region_bins=model_config['model'].get('gray_region_bins', 8),
        chroma_region_bins=model_config['model'].get('chroma_region_bins', 8),
        gray_region_temperature=model_config['model'].get('gray_region_temperature', 0.035),
        chroma_region_temperature=model_config['model'].get('chroma_region_temperature', 0.045),
        lane_detail_strength=model_config['model'].get('lane_detail_strength', 0.45),
        satellite_dependency_boost=model_config['model'].get('satellite_dependency_boost', 1.35),
        lane_evidence_threshold=model_config['model'].get('lane_evidence_threshold', 0.002),
        street_semantic_source=model_config['model'].get('street_semantic_source', 'dino_v12'),
        satellite_semantic_source=model_config['model'].get('satellite_semantic_source', 'neos'),
        dino_model_name=model_config['model'].get('dino_model_name', 'vit_small_patch16_224.dino'),
        dino_pretrained=model_config['model'].get('dino_pretrained', True),
    ).to(device)


def load_model(checkpoint_path=None):
    """Load the model."""
    global model, config, loaded_checkpoint_path
    global loaded_use_lane_vit, loaded_use_satellite_vit, loaded_use_cross_view_vit, loaded_use_semantic_cross_view_vit
    global loaded_use_semantic_color_token_match, loaded_use_polar_token_match

    if config is None:
        config = load_config(DEFAULT_CONFIG_PATH)

    checkpoint = None
    use_lane_vit = config['model'].get('use_lane_vit', False)
    use_satellite_vit = config['model'].get('use_satellite_vit', False)
    use_cross_view_vit = config['model'].get('use_cross_view_vit', False)
    use_semantic_cross_view_vit = config['model'].get('use_semantic_cross_view_vit', False)
    use_semantic_color_token_match = config['model'].get('use_semantic_color_token_match', False)
    use_polar_token_match = config['model'].get('use_polar_token_match', False)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        use_lane_vit = use_lane_vit or any(key.startswith("lane_vit.") for key in state_dict)
        use_satellite_vit = use_satellite_vit or any(key.startswith("satellite_encoder.patch_embed") for key in state_dict)
        use_cross_view_vit = use_cross_view_vit or any(key.startswith("cross_view_encoder.") for key in state_dict)
        use_semantic_cross_view_vit = use_semantic_cross_view_vit or any(key.startswith("cross_view_encoder.street_semantic_proj") for key in state_dict)
        use_semantic_color_token_match = use_semantic_color_token_match or any(key.startswith("cross_view_encoder.token_delta_mlp") for key in state_dict)

    if (
        model is None
        or loaded_use_lane_vit != use_lane_vit
        or loaded_use_satellite_vit != use_satellite_vit
        or loaded_use_cross_view_vit != use_cross_view_vit
        or loaded_use_semantic_cross_view_vit != use_semantic_cross_view_vit
        or loaded_use_semantic_color_token_match != use_semantic_color_token_match
        or loaded_use_polar_token_match != use_polar_token_match
    ):
        model_config = copy.deepcopy(config)
        model_config['model']['use_lane_vit'] = use_lane_vit
        model_config['model']['use_satellite_vit'] = use_satellite_vit
        model_config['model']['use_cross_view_vit'] = use_cross_view_vit
        model_config['model']['use_semantic_cross_view_vit'] = use_semantic_cross_view_vit
        model_config['model']['use_semantic_color_token_match'] = use_semantic_color_token_match
        model_config['model']['use_polar_token_match'] = use_polar_token_match
        model = _build_model(model_config)
        model.eval()
        loaded_checkpoint_path = None
        loaded_use_lane_vit = use_lane_vit
        loaded_use_satellite_vit = use_satellite_vit
        loaded_use_cross_view_vit = use_cross_view_vit
        loaded_use_semantic_cross_view_vit = use_semantic_cross_view_vit
        loaded_use_semantic_color_token_match = use_semantic_color_token_match
        loaded_use_polar_token_match = use_polar_token_match

    if checkpoint_path and checkpoint_path != loaded_checkpoint_path:
        print(f"[Gradio] Loading checkpoint: {checkpoint_path}", flush=True)
        if checkpoint is None:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        load_matching_state_dict(model, checkpoint['model_state_dict'])
        model.eval()
        loaded_checkpoint_path = checkpoint_path
        print(
            "[Gradio] Checkpoint loaded. "
            f"use_lane_vit={use_lane_vit}, "
            f"use_satellite_vit={use_satellite_vit}, "
            f"use_cross_view_vit={use_cross_view_vit}",
            flush=True,
        )

    return model


def rgb_to_gray(rgb):
    """Convert RGB to grayscale."""
    if len(rgb.shape) == 3:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[:, :, None].repeat(3, axis=2)
    return rgb


@torch.no_grad()
def colorize(
    street_view,
    context_image=None,
    context_type="Polar image",
    checkpoint=None,
    show_base=False,
    residual_scale=None,
    token_context="satellite",
):
    """
    Colorize street view with satellite guidance.

    Args:
        street_view: RGB or grayscale street view image
        context_image: Polar image or original satellite/overhead image
        context_type: How to interpret the context image
        checkpoint: Path to model checkpoint
        show_base: Whether to show DDColor base output

    Returns:
        Tuple of (gray_input, ddcolor_base, satellite_guided, comparison)
    """
    if street_view is None:
        return None, None, None, None, None, None, None, None, None, None

    try:
        print("[Gradio] Colorize request received.", flush=True)
        # Load model
        current_model = load_model(checkpoint)
        if residual_scale is not None and hasattr(current_model, "correction"):
            current_model.correction.residual_scale = float(residual_scale)
            current_model.residual_scale = float(residual_scale)
            print(f"[Gradio] residual_scale={float(residual_scale):.3f}", flush=True)

        # Convert to RGB
        if isinstance(street_view, np.ndarray):
            if street_view.shape[-1] == 4:  # RGBA
                street_view = street_view[:, :, :3]
        else:
            street_view = np.array(street_view)

        if context_image is None:
            raise ValueError("Please provide an overhead satellite image for context.")
        if isinstance(context_image, np.ndarray) and context_image.shape[-1] == 4:
            context_image = context_image[:, :, :3]
        elif not isinstance(context_image, np.ndarray):
            context_image = np.array(context_image)

        polar_size = tuple(config['model']['polar_input_size'])
        if context_type == "Satellite image":
            result = colorize_panorama(
                model,
                street_view.astype(np.uint8),
                context_image.astype(np.uint8),
                torch.device(device),
                polar_size,
                token_context=token_context,
            )
        else:
            gray_input, gray_patches = prepare_panorama(street_view.astype(np.uint8))
            num_patches = gray_patches.shape[0]
            polar_rgb = cv2.resize(
                context_image.astype(np.uint8),
                (polar_size[1], polar_size[0]),
                interpolation=cv2.INTER_AREA,
            )
            polar = torch.from_numpy(polar_rgb).permute(2, 0, 1).float().div(255.0)
            polar = polar.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(device)
            satellite_rgb = cv2.resize(
                context_image.astype(np.uint8),
                (256, 256),
                interpolation=cv2.INTER_AREA,
            )
            satellite = torch.from_numpy(satellite_rgb).permute(2, 0, 1).float().div(255.0)
            satellite = satellite.unsqueeze(0).repeat(num_patches, 1, 1, 1).to(device)
            patch_idx = default_patch_indices(torch.device(device), num_patches=num_patches)
            output = model(gray_patches.to(device), polar, satellite, patch_idx)

            def merge(name):
                patches = output[name].detach().cpu().permute(0, 2, 3, 1).numpy()
                return np.clip(np.concatenate(list(patches), axis=1), 0, 1)

            result = {
                "gray": gray_input,
                "polar": polar_rgb,
                "base": merge("base_rgb"),
                "final": merge("final_rgb"),
                "gray_region_vis": None,
                "satellite_chroma_vis": None,
            }
        gray_input = result['gray']
        base_rgb = (result['base'] * 255).round().astype(np.uint8)
        final_rgb = (result['final'] * 255).round().astype(np.uint8)

        # Create comparison
        comparison = np.concatenate([gray_input, base_rgb, final_rgb], axis=1)
        polar_rgb = result['polar']
        token_context_rgb = result.get('token_context_display', result.get('token_context', result.get('satellite')))
        street_semantic_vis = result.get('street_semantic')
        satellite_semantic_vis = result.get('satellite_semantic')
        gray_region_vis = result.get('gray_region_vis')
        satellite_chroma_vis = result.get('satellite_chroma_vis')

        if not show_base:
            base_rgb = None

        print("[Gradio] Colorize request finished.", flush=True)
        return (
            gray_input,
            polar_rgb,
            token_context_rgb,
            base_rgb,
            final_rgb,
            comparison,
            street_semantic_vis,
            satellite_semantic_vis,
            gray_region_vis,
            satellite_chroma_vis,
        )

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return street_view, None, None, None, None, None, None, None, None, None


def browse_dataset(split="train", sample_idx=0):
    """Browse dataset samples."""
    dataset_root = config['dataset']['root'] if config else "C:/Users/31133/Desktop/dataset1/CVUSA_processed_split"

    try:
        dataset = CVUSADataset(dataset_root, split=split, load_polar=True)

        if sample_idx >= len(dataset):
            sample_idx = 0

        sample = dataset[sample_idx]

        # Convert to uint8
        rgb = (sample['rgb'] * 255).astype(np.uint8)
        gray = (sample['gray'] * 255).astype(np.uint8)

        if sample['polar'] is not None:
            polar = (sample['polar'] * 255).astype(np.uint8)
        else:
            polar = None

        info = f"File ID: {sample['file_id']}<br>Panorama ID: {sample['panorama_id']}<br>Patch: {sample['patch_idx']}"

        return rgb, gray, polar, info, len(dataset)

    except Exception as e:
        return None, None, None, f"Error: {e}", 0


def get_available_checkpoints():
    """Get list of available checkpoints."""
    checkpoint_dir = Path("checkpoints")
    checkpoints = []

    if checkpoint_dir.exists():
        for exp_dir in checkpoint_dir.iterdir():
            if exp_dir.is_dir():
                for ckpt in exp_dir.glob("*.pth"):
                    checkpoints.append(str(ckpt))

    checkpoints = sorted(checkpoints)
    return [None] + checkpoints


def create_inference_tab():
    """Create a compact inference-only panel for the v17 model."""
    best_checkpoint = DEFAULT_CHECKPOINT_PATH

    def simple_colorize(street_view, satellite_image, residual_scale):
        (
            gray,
            polar,
            token_context_image,
            base,
            final,
            comparison,
            street_semantic,
            context_semantic,
            gray_region,
            satellite_chroma_region,
        ) = colorize(
            street_view=street_view,
            context_image=satellite_image,
            context_type="Satellite image",
            checkpoint=best_checkpoint,
            show_base=True,
            residual_scale=residual_scale,
            token_context="satellite",
        )
        if final is None:
            status = "推理失败：请检查输入图片、DDColor 权重和终端日志。"
        else:
            status = "推理完成。v17 使用 Frozen DDColor + 卫星语义颜色先验 + 语义门控 + residual correction 进行修正。"
        return final, base, comparison, street_semantic, context_semantic, token_context_image, status

    gr.Markdown(
        "## v17 测试界面 / v17 Test UI\n"
        "输入街景图和卫星图，模型会输出 DDColor 基础结果和 v17 最终结果。"
    )

    with gr.Row(equal_height=True):
        street_view_input = gr.Image(
            label="1. 街景图或全景图 / Street-view image or panorama",
            type="numpy",
            height=280,
        )
        context_input = gr.Image(
            label="2. 卫星图 / Satellite image",
            type="numpy",
            height=280,
        )

    with gr.Row():
        residual_scale_input = gr.Slider(
            label="残差修正强度 / Residual scale",
            minimum=0.0,
            maximum=2.0,
            value=1.0,
            step=0.01,
        )
        colorize_btn = gr.Button("开始上色 / Run", variant="primary", size="lg")

    status_output = gr.Markdown("等待输入。第一次运行会加载模型，可能需要 1-3 分钟。")

    with gr.Row(equal_height=True):
        final_output = gr.Image(
            label="v17 最终上色结果 / Final result",
            height=300,
        )
        base_output = gr.Image(
            label="DDColor 基础结果 / DDColor base",
            height=300,
        )

    comparison_output = gr.Image(
        label="对比图：灰度 | DDColor | v17 / Comparison: Gray | DDColor | v17",
        height=260,
    )

    with gr.Row(equal_height=True):
        street_semantic_output = gr.Image(
            label="街景语义图 / Street semantic mask",
            height=260,
        )
        satellite_semantic_output = gr.Image(
            label="卫星语义图 / Satellite semantic mask",
            height=260,
        )
        token_context_output = gr.Image(
            label="输入卫星图 / Input satellite image",
            height=260,
        )

    colorize_btn.click(
        fn=simple_colorize,
        inputs=[street_view_input, context_input, residual_scale_input],
        outputs=[
            final_output,
            base_output,
            comparison_output,
            street_semantic_output,
            satellite_semantic_output,
            token_context_output,
            status_output,
        ],
    )

    return None
def create_dataset_browser_tab():
    """Create the dataset browser tab."""
    with gr.Row():
        with gr.Column():
            split_select = gr.Dropdown(
                label="Split",
                choices=["train", "val", "test"],
                value="train",
            )
            sample_slider = gr.Slider(
                label="Sample Index",
                minimum=0,
                maximum=1000,
                value=0,
                step=1,
            )
            sample_count = gr.Number(label="Total Samples", value=0, interactive=False)
            info_text = gr.Markdown()

        with gr.Column():
            rgb_view = gr.Image(label="RGB Ground Truth")
            gray_view = gr.Image(label="Grayscale")
            polar_view = gr.Image(label="Polar Satellite View")

    def update_sample_count(split):
        dataset_root = config['dataset']['root'] if config else "C:/Users/31133/Desktop/dataset1/CVUSA_processed_split"
        try:
            dataset = CVUSADataset(dataset_root, split=split, load_polar=True)
            return len(dataset), gr.Slider(maximum=len(dataset)-1)
        except:
            return 0, gr.Slider(maximum=1000)

    def browse(split, idx):
        return browse_dataset(split, int(idx))

    split_select.change(
        fn=update_sample_count,
        inputs=[split_select],
        outputs=[sample_count, sample_slider],
    )

    sample_slider.change(
        fn=browse,
        inputs=[split_select, sample_slider],
        outputs=[rgb_view, gray_view, polar_view, info_text],
    )

    # Initial load
    rgb_view.change(
        fn=lambda x: x,
        inputs=[split_select],
        outputs=[sample_count],
    )

    return gr.Tab("Dataset Browser")


def create_training_tab():
    """Create training monitoring tab."""
    def start_training(batch, epoch_count, learning_rate):
        global training_process
        if training_process is not None and training_process.poll() is None:
            return read_training_log(), "A UI-launched training process is already running."
        base_config = load_config("config.yaml")
        run_config = copy.deepcopy(base_config)
        run_config["training"]["batch_size"] = int(batch)
        run_config["training"]["epochs"] = int(epoch_count)
        run_config["training"]["lr"] = float(learning_rate)
        exp_name = "ui_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        config_path = log_dir / f"{exp_name}.yaml"
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(run_config, file, allow_unicode=True, sort_keys=False)
        log_path = log_dir / f"{exp_name}.stdout.log"
        error_path = log_dir / f"{exp_name}.stderr.log"
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        training_process = subprocess.Popen(
            [sys.executable, "-u", "train.py", "--config", str(config_path), "--exp_name", exp_name],
            stdout=log_path.open("w", encoding="utf-8"),
            stderr=error_path.open("w", encoding="utf-8"),
            creationflags=creationflags,
        )
        return "", f"Training launched: `{exp_name}` (PID `{training_process.pid}`)"

    def read_training_log():
        logs = sorted(Path("outputs/logs").glob("*.stdout.log"), key=lambda p: p.stat().st_mtime)
        if not logs:
            return "No training log found."
        text = logs[-1].read_text(encoding="utf-8", errors="replace")
        return text[-6000:]

    with gr.Row():
        with gr.Column():
            batch_size = gr.Number(label="Batch Size", value=4)
            epochs = gr.Number(label="Epochs", value=30)
            lr = gr.Number(label="Learning Rate", value=0.0001)
            start_train_btn = gr.Button("Start Training", variant="primary")
            refresh_btn = gr.Button("Refresh Latest Log")

        with gr.Column():
            log_output = gr.Textbox(label="Training Log", lines=20, placeholder="Training logs will appear here...")
            status_output = gr.Markdown()

    start_train_btn.click(
        fn=start_training,
        inputs=[batch_size, epochs, lr],
        outputs=[log_output, status_output],
    )
    refresh_btn.click(
        fn=read_training_log,
        outputs=[log_output],
    )

    return gr.Tab("Training")


def create_evaluation_tab():
    """Create evaluation tab."""
    checkpoints = get_available_checkpoints()

    with gr.Row():
        with gr.Column():
            checkpoint_select = gr.Dropdown(
                label="Checkpoint",
                choices=checkpoints,
                value=checkpoints[0] if checkpoints else None,
            )
            split_select = gr.Dropdown(
                label="Split",
                choices=["val", "test"],
                value="val",
            )
            evaluate_btn = gr.Button("Run Evaluation", variant="primary")

        with gr.Column():
            results_output = gr.Markdown()

    evaluate_btn.click(
        fn=lambda ckpt, split: f"To run evaluation, use: python evaluate.py --checkpoint {ckpt} --split {split}",
        inputs=[checkpoint_select, split_select],
        outputs=[results_output],
    )

    return gr.Tab("Evaluation")


def create_interface():
    """Create the main Gradio interface."""
    with gr.Blocks(title="Satellite-Guided DDColor") as demo:
        gr.Markdown(
            """
            # Satellite-Guided DDColor v17

            当前权重：`satellite_constrained_v17_server/best.pth`

            流程：街景图 → Frozen DDColor → base；街景语义负责定位区域；
            卫星图和卫星语义提供颜色先验；语义门控控制哪些区域参考卫星；
            最后通过 residual correction 输出最终上色结果。天空和车辆区域主要走 street-only 分支。
            """
        )
        create_inference_tab()

    return demo
if __name__ == "__main__":
    demo = create_interface()

    # Load config for browser
    try:
        config = load_config(DEFAULT_CONFIG_PATH)
    except:
        config = None

    # Get Gradio config
    gradio_config = config.get('gradio', {}) if config else {}

    demo.launch(
        server_name=gradio_config.get('server_name', '0.0.0.0'),
        server_port=gradio_config.get('server_port', 7860),
        share=gradio_config.get('share', False),
        show_error=True,
    )


