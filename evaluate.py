"""
Evaluate model on test set.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CVUSADataset
from src.model import SatelliteGuidedDDColor
from src.metrics import MetricsCalculator
from src.utils import load_config, load_matching_state_dict


def collate_fn(batch):
    """Custom collate function."""
    gray = torch.stack([
        torch.from_numpy(b['gray']).unsqueeze(0).repeat(3, 1, 1).float() for b in batch
    ])
    rgb = torch.stack([torch.from_numpy(b['rgb']).permute(2, 0, 1).float() for b in batch])

    if batch[0]['polar'] is not None:
        polar = torch.stack([torch.from_numpy(b['polar']).permute(2, 0, 1).float() for b in batch])
    else:
        polar = None

    if batch[0].get('satellite') is not None:
        satellite = torch.stack([torch.from_numpy(b['satellite']).permute(2, 0, 1).float() for b in batch])
    else:
        satellite = None

    first_seg = next((b.get('seg') for b in batch if b.get('seg') is not None), None)
    if first_seg is not None:
        polar_seg_items = []
        for b in batch:
            seg = b.get('seg')
            if seg is None:
                seg = np.zeros_like(first_seg)
            polar_seg_items.append(torch.from_numpy(seg).permute(2, 0, 1).float())
        polar_seg = torch.stack(polar_seg_items)
    else:
        polar_seg = None

    street_semantic = None
    if batch[0].get('street_semantic') is not None:
        street_semantic = torch.stack([torch.from_numpy(b['street_semantic']).long() for b in batch])

    satellite_semantic = None
    if batch[0].get('satellite_semantic') is not None:
        satellite_semantic = torch.stack([torch.from_numpy(b['satellite_semantic']).long() for b in batch])

    return {
        'gray': gray,
        'rgb': rgb,
        'polar': polar,
        'polar_seg': polar_seg,
        'satellite': satellite,
        'street_semantic': street_semantic,
        'satellite_semantic': satellite_semantic,
        'patch_idx': torch.tensor([b['patch_idx'] for b in batch], dtype=torch.long),
        'file_id': [b['file_id'] for b in batch],
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Evaluate Satellite-Guided DDColor')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--output', type=str, default='outputs/evaluation_results.json')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--num_samples', type=int, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = SatelliteGuidedDDColor(
        ddcolor_weights_path=config['ddcolor']['weights_path'],
        ddcolor_code_path=config.get('ddcolor', {}).get('code_path'),
        context_dim=config['model']['context_dim'],
        polar_encoder_pretrained=config['model']['polar_encoder_pretrained'],
        satellite_encoder_pretrained=config['model'].get('satellite_encoder_pretrained', True),
        correction_type=config['model']['correction_type'],
        residual_scale=config['model']['residual_scale'],
        use_polar_context=config['model'].get('use_polar_context', True),
        use_lane_vit=config['model'].get('use_lane_vit', False),
        lane_vit_embed_dim=config['model'].get('lane_vit_embed_dim', 192),
        lane_vit_depth=config['model'].get('lane_vit_depth', 4),
        lane_vit_heads=config['model'].get('lane_vit_heads', 3),
        lane_vit_patch_size=config['model'].get('lane_vit_patch_size', 16),
        lane_feature_dim=config['model'].get('lane_feature_dim', 64),
        use_satellite_vit=config['model'].get('use_satellite_vit', False),
        satellite_vit_embed_dim=config['model'].get('satellite_vit_embed_dim', 192),
        satellite_vit_depth=config['model'].get('satellite_vit_depth', 4),
        satellite_vit_heads=config['model'].get('satellite_vit_heads', 3),
        satellite_vit_patch_size=config['model'].get('satellite_vit_patch_size', 16),
        satellite_vit_feature_dim=config['model'].get('satellite_vit_feature_dim', 64),
        use_cross_view_vit=config['model'].get('use_cross_view_vit', False),
        cross_view_embed_dim=config['model'].get('cross_view_embed_dim', 192),
        cross_view_depth=config['model'].get('cross_view_depth', 3),
        cross_view_heads=config['model'].get('cross_view_heads', 3),
        use_semantic_cross_view_vit=config['model'].get('use_semantic_cross_view_vit', False),
        use_semantic_color_token_match=config['model'].get('use_semantic_color_token_match', False),
        use_polar_token_match=config['model'].get('use_polar_token_match', False),
        use_semantic_cnn_context=config['model'].get('use_semantic_cnn_context', False),
        semantic_num_classes=config['model'].get('semantic_num_classes', 6),
        cross_view_patch_size=config['model'].get('cross_view_patch_size', 16),
        cross_view_street_patch_size=config['model'].get('cross_view_street_patch_size', config['model'].get('cross_view_patch_size', 16)),
        cross_view_satellite_patch_size=config['model'].get('cross_view_satellite_patch_size', 8),
        cross_view_feature_dim=config['model'].get('cross_view_feature_dim', 64),
        color_token_match_weight=config['model'].get('color_token_match_weight', 3.0),
        semantic_distribution_weight=config['model'].get('semantic_distribution_weight', 2.0),
        boundary_match_weight=config['model'].get('boundary_match_weight', 2.0),
        token_delta_scale=config['model'].get('token_delta_scale', 0.35),
        token_correction_scale=config['model'].get('token_correction_scale', 0.8),
        street_object_hidden_dim=config['model'].get('street_object_hidden_dim', 96),
        street_object_num_masks=config['model'].get('street_object_num_masks', 8),
        street_object_detail_scale=config['model'].get('street_object_detail_scale', 0.18),
        satellite_prior_strength=config['model'].get('satellite_prior_strength', 0.65),
        use_street_gray_edges=config['model'].get('use_street_gray_edges', False),
        use_street_gray_modulation=config['model'].get('use_street_gray_modulation', True),
        use_gray_satellite_token_selection=config['model'].get('use_gray_satellite_token_selection', True),
        use_satellite_chroma_token_selection=config['model'].get('use_satellite_chroma_token_selection', False),
        token_selection_patch_size=config['model'].get('token_selection_patch_size', 16),
        token_selection_dim=config['model'].get('token_selection_dim', 32),
        lane_detail_strength=config['model'].get('lane_detail_strength', 0.45),
        satellite_dependency_boost=config['model'].get('satellite_dependency_boost', 1.35),
        lane_evidence_threshold=config['model'].get('lane_evidence_threshold', 0.002),
        street_semantic_source=config['model'].get('street_semantic_source', 'dino_v12'),
        satellite_semantic_source=config['model'].get('satellite_semantic_source', 'neos'),        dino_model_name=config['model'].get('dino_model_name', 'vit_small_patch16_224.dino'),
        dino_pretrained=config['model'].get('dino_pretrained', True),
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_matching_state_dict(model, checkpoint['model_state_dict'])

    model.eval()

    # Load dataset
    print(f"Loading {args.split} set...")
    semantic_cfg = config.get('semantic', {})
    use_street_object_aware = config['model'].get('correction_type') == 'street_object_aware'
    load_semantics = bool(
        (
            config['model'].get('use_semantic_cross_view_vit', False)
            or config['model'].get('use_semantic_color_token_match', False)
            or config['model'].get('use_polar_token_match', False)
            or config['model'].get('use_semantic_cnn_context', False)
            or use_street_object_aware
        )
        and semantic_cfg.get('load_precomputed', False)
    )
    dataset = CVUSADataset(
        config['dataset']['root'],
        split=args.split,
        load_polar=True,
        use_segmap=use_street_object_aware,
        load_semantics=load_semantics,
        street_semantic_dirname=semantic_cfg.get('street_dirname', 'street_semantic'),
        satellite_semantic_dirname=semantic_cfg.get('satellite_dirname', 'overhead_satellite_semantic'),
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No paired samples are available in split '{args.split}'. "
            "Use --split val until the test files are repaired or regenerated."
        )

    if args.num_samples:
        dataset = torch.utils.data.Subset(dataset, list(range(args.num_samples)))

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Metrics calculator
    base_calc = MetricsCalculator(device=device)
    final_calc = MetricsCalculator(device=device)

    # Results storage
    results = {
        'checkpoint': args.checkpoint,
        'split': args.split,
        'num_samples': len(dataset),
        'base_metrics': {},
        'final_metrics': {},
    }

    # Evaluate
    print("\nEvaluating...")

    for batch in tqdm(dataloader):
        gray = batch['gray'].to(device)
        rgb_gt = batch['rgb'].to(device)
        polar = batch['polar'].to(device)
        polar_seg = batch.get('polar_seg')
        if polar_seg is not None:
            polar_seg = polar_seg.to(device)
        satellite = batch['satellite'].to(device)
        patch_idx = batch['patch_idx'].to(device)
        street_semantic = batch.get('street_semantic')
        satellite_semantic = batch.get('satellite_semantic')
        if street_semantic is not None:
            street_semantic = street_semantic.to(device)
        if satellite_semantic is not None:
            satellite_semantic = satellite_semantic.to(device)

        # Resize polar if needed
        polar_h, polar_w = polar.shape[2:]
        target_h, target_w = config['model']['polar_input_size']
        if polar_h != target_h or polar_w != target_w:
            import torch.nn.functional as F
            polar = F.interpolate(polar, size=(target_h, target_w), mode='bilinear')
        sat_h, sat_w = satellite.shape[2:]
        if sat_h != 256 or sat_w != 256:
            import torch.nn.functional as F
            satellite = F.interpolate(satellite, size=(256, 256), mode='bilinear')

        # Forward
        output = model(
            gray,
            polar,
            satellite,
            patch_idx,
            polar_seg=polar_seg,
            street_semantic=street_semantic,
            satellite_semantic=satellite_semantic,
        )

        # Compute metrics for base
        base_metrics = base_calc.compute_batch(output['base_rgb'], rgb_gt, accumulate_fid=True)

        # Compute metrics for final
        final_metrics = final_calc.compute_batch(output['final_rgb'], rgb_gt, accumulate_fid=True)

        # Accumulate
        for k, v in base_metrics.items():
            if v is not None and v != float('inf'):
                results['base_metrics'].setdefault(k, []).append(v)

        for k, v in final_metrics.items():
            if v is not None and v != float('inf'):
                results['final_metrics'].setdefault(k, []).append(v)

    # Compute averages
    for key in results['final_metrics']:
        results['final_metrics'][key] = float(np.mean(results['final_metrics'][key]))
    for key in results['base_metrics']:
        results['base_metrics'][key] = float(np.mean(results['base_metrics'][key]))

    base_fid = base_calc.get_fid()
    final_fid = final_calc.get_fid()
    if base_fid is not None:
        results['base_metrics']['fid'] = float(base_fid)
    if final_fid is not None:
        results['final_metrics']['fid'] = float(final_fid)

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(dataset)}")
    print()

    print("DDColor Base:")
    for k, v in results['base_metrics'].items():
        print(f"  {k}: {v:.4f}")

    print("\nSatellite-Guided (Final):")
    for k, v in results['final_metrics'].items():
        print(f"  {k}: {v:.4f}")

    print("\nImprovement:")
    for k in results['base_metrics']:
        base = results['base_metrics'].get(k, 0)
        final = results['final_metrics'].get(k, 0)
        if k in ('lpips', 'fid'):
            improv = base - final  # Lower is better for LPIPS
        else:
            improv = final - base
        print(f"  {k}: {improv:+.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()


