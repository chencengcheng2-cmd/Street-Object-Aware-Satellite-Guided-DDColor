"""Inference for four-patch satellite-guided street-view colorization."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model import SatelliteGuidedDDColor
from src.polar_transform import create_polar_from_satellite
from src.utils import load_config, load_matching_state_dict

SEGFORMER_PROCESSOR = None
SEGFORMER_MODEL = None
SEGFORMER_MAPPING = None
SEGFORMER_DEVICE = None
DEEPLAB_SAT_MODEL = None
DEEPLAB_SAT_CHECKPOINT = None
DEEPLAB_SAT_DEVICE = None
DEEPLAB_SAT_CLASSES = ["road", "building", "grass", "tree", "car", "other"]
DEEPLAB_SAT_TO_NEOS = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)
DEEPLAB_SAT_OTHER_THRESHOLD = 1.01
DEFAULT_DEEPLAB_SAT_CHECKPOINT = "checkpoints/satellite_segmentation_v13/best.pth"

NEOS_PALETTE = np.array(
    [
        [135, 206, 235],  # sky
        [255, 255, 255],  # impervious / roads
        [0, 0, 255],      # building
        [0, 255, 255],    # low vegetation
        [0, 128, 0],      # tree
        [255, 255, 0],    # car
        [255, 0, 0],      # clutter / background
    ],
    dtype=np.uint8,
)

REGION_PALETTE = np.array(
    [
        [230, 57, 70],
        [244, 162, 97],
        [233, 196, 106],
        [42, 157, 143],
        [38, 70, 83],
        [69, 123, 157],
        [168, 218, 220],
        [131, 56, 236],
    ],
    dtype=np.uint8,
)


def load_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def is_panorama_image(street_rgb: np.ndarray, aspect_threshold: float = 2.5) -> bool:
    """Return True when an input should be treated as a wide panorama."""
    h, w = street_rgb.shape[:2]
    return (w / max(h, 1)) >= aspect_threshold


def normalize_street_image(street_rgb: np.ndarray) -> tuple[np.ndarray, int]:
    """Resize input to either one 256x256 patch or four panorama patches."""
    if is_panorama_image(street_rgb):
        return cv2.resize(street_rgb, (1024, 256), interpolation=cv2.INTER_AREA), 4
    return cv2.resize(street_rgb, (256, 256), interpolation=cv2.INTER_AREA), 1


def prepare_panorama(street_rgb: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
    """Normalize an input street image to one or four 256x256 grayscale patches."""
    resized, num_patches = normalize_street_image(street_rgb)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray_rgb = np.repeat(gray[:, :, None], 3, axis=2)
    patches = np.stack([gray_rgb[:, i * 256:(i + 1) * 256] for i in range(num_patches)])
    tensor = torch.from_numpy(patches).permute(0, 3, 1, 2).float() / 255.0
    return gray_rgb, tensor


def default_patch_indices(device: torch.device = None, num_patches: int = 4) -> torch.Tensor:
    """Return direction IDs. Index 0 means non-panorama or unknown direction."""
    if num_patches == 4:
        patch_idx = torch.arange(1, 5, dtype=torch.long)
    else:
        patch_idx = torch.zeros(num_patches, dtype=torch.long)
    return patch_idx.to(device) if device is not None else patch_idx


def _load_segformer_neos(device: torch.device, model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512"):
    """Lazy-load the SegFormer pseudo-NEOS segmenter used for single-image inference."""
    global SEGFORMER_PROCESSOR, SEGFORMER_MODEL, SEGFORMER_MAPPING, SEGFORMER_DEVICE
    if SEGFORMER_MODEL is not None and SEGFORMER_DEVICE == str(device):
        return SEGFORMER_PROCESSOR, SEGFORMER_MODEL, SEGFORMER_MAPPING

    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    from scripts.generate_neos_segformer_semantics import build_id_mapping

    print(f"[Inference] Loading SegFormer semantic model on {device}...", flush=True)
    SEGFORMER_PROCESSOR = SegformerImageProcessor.from_pretrained(model_name)
    SEGFORMER_MODEL = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device).eval()
    SEGFORMER_MAPPING = build_id_mapping(SEGFORMER_MODEL.config.id2label)
    SEGFORMER_DEVICE = str(device)
    return SEGFORMER_PROCESSOR, SEGFORMER_MODEL, SEGFORMER_MAPPING


@torch.inference_mode()
def generate_neos_semantic(
    rgb: np.ndarray,
    is_satellite: bool,
    device: torch.device,
) -> np.ndarray:
    """Generate a 7-class NEOS-style semantic label map for one RGB image."""
    from scripts.generate_neos_segformer_semantics import apply_neos_rules

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    processor, segmenter, mapping = _load_segformer_neos(device)
    h, w = rgb.shape[:2]
    inputs = processor(images=rgb, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logits = segmenter(**inputs).logits
    logits = torch.nn.functional.interpolate(
        logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    raw = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
    raw = np.clip(raw, 0, len(mapping) - 1)
    labels = mapping[raw].copy()
    labels = apply_neos_rules(rgb, labels, is_satellite=is_satellite)
    return labels.astype(np.uint8)


def colorize_neos_labels(labels: np.ndarray) -> np.ndarray:
    """Convert NEOS class IDs to an RGB visualization."""
    labels = np.asarray(labels).astype(np.int64)
    labels = np.clip(labels, 0, len(NEOS_PALETTE) - 1)
    return NEOS_PALETTE[labels]


def colorize_region_labels(labels: np.ndarray) -> np.ndarray:
    """Convert v20 gray/chroma region IDs to an RGB visualization."""
    labels = np.asarray(labels).astype(np.int64)
    labels = labels % len(REGION_PALETTE)
    return REGION_PALETTE[labels]


class _SegmentationModelWrapper(torch.nn.Module):
    """Return the logits tensor from torchvision segmentation models."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            return out["out"]
        return out


def _load_deeplab_satellite_model(
    device: torch.device,
    checkpoint_path: str = DEFAULT_DEEPLAB_SAT_CHECKPOINT,
) -> tuple[torch.nn.Module, list[str]]:
    """Lazy-load the custom DeepLabV3 satellite segmenter."""
    global DEEPLAB_SAT_MODEL, DEEPLAB_SAT_CHECKPOINT, DEEPLAB_SAT_DEVICE, DEEPLAB_SAT_CLASSES
    checkpoint_path = str(Path(checkpoint_path))
    if (
        DEEPLAB_SAT_MODEL is not None
        and DEEPLAB_SAT_CHECKPOINT == checkpoint_path
        and DEEPLAB_SAT_DEVICE == str(device)
    ):
        return DEEPLAB_SAT_MODEL, DEEPLAB_SAT_CLASSES

    from torchvision.models.segmentation import deeplabv3_resnet50

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(__file__).resolve().parent / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing DeepLabV3 satellite checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = checkpoint.get("classes", DEEPLAB_SAT_CLASSES)
    model_inner = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        num_classes=len(classes),
        aux_loss=False,
    )
    model = _SegmentationModelWrapper(model_inner).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    DEEPLAB_SAT_MODEL = model
    DEEPLAB_SAT_CHECKPOINT = checkpoint_path
    DEEPLAB_SAT_DEVICE = str(device)
    DEEPLAB_SAT_CLASSES = list(classes)
    print(f"[Inference] Loaded DeepLabV3 satellite segmenter: {ckpt_path}", flush=True)
    return model, DEEPLAB_SAT_CLASSES


@torch.inference_mode()
def generate_deeplab_satellite_semantic(
    rgb: np.ndarray,
    device: torch.device,
    checkpoint_path: str = DEFAULT_DEEPLAB_SAT_CHECKPOINT,
    other_threshold: float = DEEPLAB_SAT_OTHER_THRESHOLD,
) -> np.ndarray:
    """Generate NEOS-compatible satellite labels with the custom DeepLabV3 model.

    DeepLabV3 classes are [road, building, grass, tree, car, other]. The color
    correction model expects the 7-class NEOS order:
    [sky, road, building, low vegetation, tree, car, other]. Satellite images
    have no sky. The satellite-side "other" class is removed by replacing it
    with the strongest non-other prediction before mapping to NEOS ids.
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    model, classes = _load_deeplab_satellite_model(device, checkpoint_path)
    h, w = rgb.shape[:2]
    image_256 = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image_256).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    logits = model(tensor)[0]
    pred = logits.argmax(dim=0)
    if "other" in classes and len(classes) > 1:
        other_idx = classes.index("other")
        probs = logits.softmax(dim=0)
        non_other_logits = logits.clone()
        non_other_logits[other_idx] = -1e9
        second_choice = non_other_logits.argmax(dim=0)
        replace_mask = (pred == other_idx) & (probs[other_idx] < float(other_threshold))
        pred = torch.where(replace_mask, second_choice, pred)
    pred_np = pred.detach().cpu().numpy().astype(np.uint8)
    mapped = DEEPLAB_SAT_TO_NEOS[np.clip(pred_np, 0, len(DEEPLAB_SAT_TO_NEOS) - 1)]
    if (h, w) != (256, 256):
        mapped = cv2.resize(mapped, (w, h), interpolation=cv2.INTER_NEAREST)
    return mapped.astype(np.uint8)


def build_online_semantics(
    model: SatelliteGuidedDDColor,
    street_rgb: np.ndarray,
    context_rgb_256: np.ndarray,
    device: torch.device,
    context_labels_256: np.ndarray | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Create street and context semantic tensors for semantic models.

    Street semantics follow the same patch path as colorization. Wide panoramas
    are resized to 1024x256, split into four 256x256 patches first, then each
    patch is segmented independently. Normal images are resized to one 256x256
    patch and segmented once. Context semantics are shared by all street patches.
    For Polar mode, the caller can segment the normal-format polar map first and
    pass a nearest-neighbor resized 256x256 label map through context_labels_256.
    """
    needs_semantics = bool(
        getattr(model, "use_semantic_cnn_context", False)
        or getattr(model, "use_semantic_cross_view_vit", False)
        or getattr(model, "use_semantic_color_token_match", False)
        or getattr(model, "use_polar_token_match", False)
        or getattr(model, "use_street_object_aware", False)
        or getattr(model, "use_satellite_color_bottleneck", False)
        or getattr(model, "use_satellite_detail_bottleneck", False)
        or getattr(model, "use_sky_gray_chroma_match", False)
    )
    if not needs_semantics:
        return None, None

    street_resized, num_patches = normalize_street_image(street_rgb)
    street_rgb_patches = [
        street_resized[:, i * 256:(i + 1) * 256]
        for i in range(num_patches)
    ]
    street_patches = np.stack([
        generate_neos_semantic(patch, is_satellite=False, device=device)
        for patch in street_rgb_patches
    ])
    street_semantic = torch.from_numpy(street_patches).long().to(device)

    if context_labels_256 is None:
        satellite_labels = generate_deeplab_satellite_semantic(context_rgb_256, device=device)
    else:
        satellite_labels = context_labels_256.astype(np.uint8)
    satellite_semantic = torch.from_numpy(satellite_labels).long().unsqueeze(0).repeat(num_patches, 1, 1).to(device)
    return street_semantic, satellite_semantic


def harmonize_panorama_patches_lab(
    patches: np.ndarray,
    labels: np.ndarray | None,
    satellite_prior: np.ndarray | None = None,
    classes: tuple[int, ...] = (1, 2, 3, 4),
    alpha: float = 0.45,
    satellite_weight: float = 0.70,
    min_pixels: int = 512,
    max_ab_shift: float = 18.0,
) -> np.ndarray:
    """Reduce color seams across panorama patches in semantic Lab space.

    The operation keeps Lab L unchanged and only shifts a/b chroma channels.
    It is intentionally semantic-aware:
    - road/building/grass/tree are harmonized separately;
    - sky/car/other are left unchanged;
    - satellite color prior is used when available, so harmonization still
      follows satellite guidance instead of a blind global white balance.
    """
    if labels is None or patches.ndim != 4 or patches.shape[0] <= 1:
        return patches
    if labels.shape[0] != patches.shape[0] or labels.shape[-2:] != patches.shape[1:3]:
        return patches

    labs = np.stack([
        cv2.cvtColor(np.clip(patch, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2LAB)
        for patch in patches
    ])
    prior_labs = None
    if satellite_prior is not None and satellite_prior.shape == patches.shape:
        prior_labs = np.stack([
            cv2.cvtColor(np.clip(patch, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2LAB)
            for patch in satellite_prior
        ])

    for class_id in classes:
        masks = labels == class_id
        total_pixels = int(masks.sum())
        if total_pixels < min_pixels:
            continue

        output_ab = labs[:, :, :, 1:3][masks].reshape(-1, 2).mean(axis=0)
        target_ab = output_ab
        if prior_labs is not None:
            prior_pixels = prior_labs[:, :, :, 1:3][masks].reshape(-1, 2)
            if prior_pixels.shape[0] >= min_pixels:
                prior_ab = prior_pixels.mean(axis=0)
                target_ab = satellite_weight * prior_ab + (1.0 - satellite_weight) * output_ab

        for patch_idx in range(patches.shape[0]):
            mask = masks[patch_idx]
            if int(mask.sum()) < min_pixels:
                continue
            current_ab = labs[patch_idx, :, :, 1:3][mask].reshape(-1, 2).mean(axis=0)
            shift = alpha * (target_ab - current_ab)
            shift = np.clip(shift, -max_ab_shift, max_ab_shift)

            # Soft mask avoids visible chroma discontinuities at semantic borders.
            weight = mask.astype(np.float32)
            weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=2.0, sigmaY=2.0)
            weight = np.clip(weight, 0.0, 1.0)
            labs[patch_idx, :, :, 1] += weight * shift[0]
            labs[patch_idx, :, :, 2] += weight * shift[1]

    harmonized = np.stack([
        cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
        for lab in labs
    ])
    return np.clip(harmonized, 0.0, 1.0)


@torch.inference_mode()
def colorize_panorama(
    model: SatelliteGuidedDDColor,
    street_rgb: np.ndarray,
    satellite_rgb: np.ndarray,
    device: torch.device,
    polar_size: tuple[int, int] = (256, 512),
    use_online_semantics: bool = True,
    token_context: str = "satellite",
    use_panorama_harmonization: bool = True,
) -> dict:
    gray_rgb, gray_patches = prepare_panorama(street_rgb)
    num_patches = gray_patches.shape[0]
    needs_polar = (
        token_context == "polar"
        or getattr(model, "polar_encoder", None) is not None
        or getattr(model, "use_street_object_aware", False)
    )
    polar_rgb = None
    polar_rgb_for_semantic = None
    if needs_polar:
        polar_rgb = create_polar_from_satellite(satellite_rgb, output_size=polar_size)
        if getattr(model, "use_street_object_aware", False):
            semantic_polar_size = (polar_size[0], max(polar_size[1] * 2, 1024))
            polar_rgb_for_semantic = create_polar_from_satellite(
                satellite_rgb,
                output_size=semantic_polar_size,
            )
        else:
            polar_rgb_for_semantic = polar_rgb
        polar = torch.from_numpy(polar_rgb).permute(2, 0, 1).float().div(255.0)
        polar = polar.unsqueeze(0).repeat(num_patches, 1, 1, 1)
    else:
        # v8 has use_polar_context=False, so this tensor is only an API placeholder.
        polar = torch.zeros(num_patches, 3, polar_size[0], polar_size[1], dtype=torch.float32)
    satellite_resized = cv2.resize(satellite_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    token_context_rgb = satellite_resized
    token_context_display = satellite_resized
    token_context_semantic_display_labels = None
    token_context_labels_256 = None
    polar_seg = None
    if token_context == "polar":
        # Test mode for v8: use the generated polar image as the token context.
        # The visible polar map keeps the CVUSA normal format (512x128, W x H).
        # The semantic mask is generated on that normal-format polar map first,
        # then resized with nearest-neighbor interpolation for the v8 token matcher.
        token_context_display = polar_rgb_for_semantic
        token_context_rgb = cv2.resize(polar_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    elif getattr(model, "use_street_object_aware", False):
        # v12 was trained with satellite-side semantic masks. Keep the token
        # context and the online semantic mask in the satellite coordinate frame.
        token_context_display = satellite_resized
    satellite = torch.from_numpy(token_context_rgb).permute(2, 0, 1).float().div(255.0)
    satellite = satellite.unsqueeze(0).repeat(num_patches, 1, 1, 1)
    patch_idx = default_patch_indices(device, num_patches=num_patches)
    street_semantic = None
    satellite_semantic = None
    if use_online_semantics:
        if token_context == "polar":
            token_context_semantic_display_labels = generate_neos_semantic(
                polar_rgb,
                is_satellite=True,
                device=device,
            )
            token_context_labels_256 = cv2.resize(
                token_context_semantic_display_labels,
                (256, 256),
                interpolation=cv2.INTER_NEAREST,
            )
        elif (
            getattr(model, "use_street_object_aware", False)
            or getattr(model, "use_satellite_color_bottleneck", False)
            or getattr(model, "use_satellite_detail_bottleneck", False)
            or getattr(model, "use_sky_gray_chroma_match", False)
        ):
            token_context_semantic_display_labels = generate_deeplab_satellite_semantic(
                satellite_resized,
                device=device,
            )
            token_context_labels_256 = token_context_semantic_display_labels
        street_semantic, satellite_semantic = build_online_semantics(
            model,
            street_rgb,
            token_context_rgb,
            device,
            context_labels_256=token_context_labels_256,
        )
    output = model(
        gray_patches.to(device),
        polar.to(device),
        satellite.to(device),
        patch_idx,
        polar_seg=polar_seg.to(device) if polar_seg is not None else None,
        street_semantic=street_semantic,
        satellite_semantic=satellite_semantic,
    )

    def get_patches(name: str) -> np.ndarray:
        patches = output[name].detach().cpu().permute(0, 2, 3, 1).numpy()
        return np.clip(patches, 0, 1)

    def merge_patches(patches: np.ndarray) -> np.ndarray:
        return np.clip(np.concatenate(list(patches), axis=1), 0, 1)

    base_patches = get_patches("base_rgb")
    final_patches = get_patches("final_rgb")
    satellite_prior_patches = None
    if isinstance(output.get("satellite_color_prior"), torch.Tensor):
        satellite_prior_patches = get_patches("satellite_color_prior")

    gray_region_vis = None
    if isinstance(output.get("gray_region_probs"), torch.Tensor):
        gray_probs = output["gray_region_probs"].detach().cpu().numpy()
        gray_labels = gray_probs.argmax(axis=1)
        gray_labels_merged = np.concatenate(list(gray_labels), axis=1)
        gray_region_vis = colorize_region_labels(gray_labels_merged)
        if isinstance(output.get("sky_mask"), torch.Tensor):
            sky_mask = output["sky_mask"].detach().cpu().numpy()
            if sky_mask.ndim == 4:
                sky_mask = sky_mask[:, 0]
            sky_mask_merged = np.concatenate(list(sky_mask), axis=1) > 0.5
            gray_region_vis[sky_mask_merged] = NEOS_PALETTE[0]

    satellite_chroma_vis = None
    if isinstance(output.get("satellite_chroma_regions"), torch.Tensor):
        chroma_probs = output["satellite_chroma_regions"].detach().cpu().numpy()
        if chroma_probs.ndim == 4:
            chroma_labels = chroma_probs[0].argmax(axis=0)
        else:
            chroma_labels = chroma_probs.argmax(axis=0)
        satellite_chroma_vis = colorize_region_labels(chroma_labels)

    street_labels = None
    if street_semantic is not None:
        street_labels = street_semantic.detach().cpu().numpy()
    if use_panorama_harmonization and num_patches > 1 and street_labels is not None:
        final_patches = harmonize_panorama_patches_lab(
            final_patches,
            street_labels,
            satellite_prior=satellite_prior_patches,
        )

    base_rgb = merge_patches(base_patches)
    final_rgb = merge_patches(final_patches)
    street_semantic_vis = None
    satellite_semantic_vis = None
    street_labels_merged = None
    if street_semantic is not None:
        street_labels_merged = np.concatenate(list(street_labels), axis=1)
        street_semantic_vis = colorize_neos_labels(street_labels_merged)
    if satellite_semantic is not None:
        if token_context_semantic_display_labels is not None:
            satellite_semantic_vis = colorize_neos_labels(token_context_semantic_display_labels)
        else:
            satellite_labels = satellite_semantic[0].detach().cpu().numpy()
            satellite_semantic_vis = colorize_neos_labels(satellite_labels)

    return {
        "gray": gray_rgb,
        "polar": polar_rgb,
        "satellite": satellite_resized,
        "token_context": token_context_rgb,
        "token_context_display": token_context_display,
        "street_semantic": street_semantic_vis,
        "satellite_semantic": satellite_semantic_vis,
        "gray_region_vis": gray_region_vis,
        "satellite_chroma_vis": satellite_chroma_vis,
        "base": base_rgb,
        "final": final_rgb,
    }


def save_rgb_float(image: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = (np.clip(image, 0, 1) * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Inference with Satellite-Guided DDColor")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--street_view", required=True)
    parser.add_argument("--satellite", required=True)
    parser.add_argument("--output", default="outputs/inference/output.jpg")
    parser.add_argument("--show_base", action="store_true")
    parser.add_argument(
        "--no_harmonize",
        action="store_true",
        help="Disable semantic Lab panorama color harmonization.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SatelliteGuidedDDColor(
        ddcolor_weights_path=config["ddcolor"]["weights_path"],
        ddcolor_code_path=config["ddcolor"].get("code_path"),
        context_dim=config["model"]["context_dim"],
        polar_encoder_pretrained=config["model"]["polar_encoder_pretrained"],
        satellite_encoder_pretrained=config["model"].get("satellite_encoder_pretrained", True),
        correction_type=config["model"]["correction_type"],
        residual_scale=config["model"]["residual_scale"],
        use_polar_context=config["model"].get("use_polar_context", True),
        use_lane_vit=config["model"].get("use_lane_vit", False),
        lane_vit_embed_dim=config["model"].get("lane_vit_embed_dim", 192),
        lane_vit_depth=config["model"].get("lane_vit_depth", 4),
        lane_vit_heads=config["model"].get("lane_vit_heads", 3),
        lane_vit_patch_size=config["model"].get("lane_vit_patch_size", 16),
        lane_feature_dim=config["model"].get("lane_feature_dim", 64),
        use_satellite_vit=config["model"].get("use_satellite_vit", False),
        satellite_vit_embed_dim=config["model"].get("satellite_vit_embed_dim", 192),
        satellite_vit_depth=config["model"].get("satellite_vit_depth", 4),
        satellite_vit_heads=config["model"].get("satellite_vit_heads", 3),
        satellite_vit_patch_size=config["model"].get("satellite_vit_patch_size", 16),
        satellite_vit_feature_dim=config["model"].get("satellite_vit_feature_dim", 64),
        use_cross_view_vit=config["model"].get("use_cross_view_vit", False),
        cross_view_embed_dim=config["model"].get("cross_view_embed_dim", 192),
        cross_view_depth=config["model"].get("cross_view_depth", 3),
        cross_view_heads=config["model"].get("cross_view_heads", 3),
        use_semantic_cross_view_vit=config["model"].get("use_semantic_cross_view_vit", False),
        use_semantic_color_token_match=config["model"].get("use_semantic_color_token_match", False),
        use_polar_token_match=config["model"].get("use_polar_token_match", False),
        use_semantic_cnn_context=config["model"].get("use_semantic_cnn_context", False),
        semantic_num_classes=config["model"].get("semantic_num_classes", 6),
        cross_view_patch_size=config["model"].get("cross_view_patch_size", 16),
        cross_view_street_patch_size=config["model"].get("cross_view_street_patch_size", config["model"].get("cross_view_patch_size", 16)),
        cross_view_satellite_patch_size=config["model"].get("cross_view_satellite_patch_size", 8),
        cross_view_feature_dim=config["model"].get("cross_view_feature_dim", 64),
        color_token_match_weight=config["model"].get("color_token_match_weight", 3.0),
        semantic_distribution_weight=config["model"].get("semantic_distribution_weight", 2.0),
        boundary_match_weight=config["model"].get("boundary_match_weight", 2.0),
        token_delta_scale=config["model"].get("token_delta_scale", 0.35),
        token_correction_scale=config["model"].get("token_correction_scale", 0.8),
        street_object_hidden_dim=config["model"].get("street_object_hidden_dim", 96),
        street_object_num_masks=config["model"].get("street_object_num_masks", 8),
        street_object_detail_scale=config["model"].get("street_object_detail_scale", 0.18),
        satellite_prior_strength=config["model"].get("satellite_prior_strength", 0.65),
        use_street_gray_edges=config["model"].get("use_street_gray_edges", False),
        use_street_gray_modulation=config["model"].get("use_street_gray_modulation", True),
        use_gray_satellite_token_selection=config["model"].get("use_gray_satellite_token_selection", True),
            use_satellite_chroma_token_selection=config["model"].get("use_satellite_chroma_token_selection", False),
        token_selection_patch_size=config["model"].get("token_selection_patch_size", 16),
        token_selection_dim=config["model"].get("token_selection_dim", 32),
        lane_detail_strength=config["model"].get("lane_detail_strength", 0.45),
        satellite_dependency_boost=config["model"].get("satellite_dependency_boost", 1.35),
        lane_evidence_threshold=config["model"].get("lane_evidence_threshold", 0.002),
        street_semantic_source=config["model"].get("street_semantic_source", "dino_v12"),
        satellite_semantic_source=config["model"].get("satellite_semantic_source", "neos"),
        dino_model_name=config["model"].get("dino_model_name", "vit_small_patch16_224.dino"),
        dino_pretrained=config["model"].get("dino_pretrained", True),
        device=str(device),
    )
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        load_matching_state_dict(model, checkpoint["model_state_dict"])
    model.eval()

    result = colorize_panorama(
        model,
        load_rgb(args.street_view),
        load_rgb(args.satellite),
        device,
        tuple(config["model"]["polar_input_size"]),
        use_panorama_harmonization=not args.no_harmonize,
    )
    output_path = Path(args.output)
    save_rgb_float(result["final"], output_path)
    if args.show_base:
        save_rgb_float(result["base"], output_path.with_name(output_path.stem + "_base" + output_path.suffix))
    polar_path = output_path.with_name(output_path.stem + "_polar" + output_path.suffix)
    if result.get("polar") is not None:
        cv2.imwrite(str(polar_path), cv2.cvtColor(result["polar"], cv2.COLOR_RGB2BGR))
    print(f"Final result: {output_path}")
    if result.get("polar") is not None:
        print(f"Polar context: {polar_path}")


if __name__ == "__main__":
    main()

