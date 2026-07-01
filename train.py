"""
Training script for Satellite-Guided DDColor.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import CVUSADataset, create_dataset_report
from src.model import SatelliteGuidedDDColor
from src.trainer import Trainer
from src.utils import load_config, load_matching_state_dict, set_seed, create_timestamp_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train Satellite-Guided DDColor')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no_ddcolor', action='store_true',
                        help='Skip DDColor and use dummy model for testing')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Run a short smoke test')
    parser.add_argument('--quick_train', action='store_true',
                        help='Run two train and two validation batches to verify the full pipeline')
    return parser.parse_args()


def collate_fn(batch):
    """Custom collate function for the dataset."""
    gray = torch.stack([
        torch.from_numpy(b['gray']).unsqueeze(2).repeat(1, 1, 3).permute(2, 0, 1).float()
        for b in batch
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
        'panorama_id': [b['panorama_id'] for b in batch],
    }


class PanoramaBatchSampler:
    """Batch sampler that keeps the four patches of each panorama together."""

    def __init__(
        self,
        dataset: CVUSADataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = max(4, int(batch_size))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.panoramas_per_batch = max(1, self.batch_size // 4)

        grouped = {}
        for index, sample in enumerate(dataset.samples):
            grouped.setdefault(sample["panorama_id"], []).append((sample["patch_idx"], index))
        self.groups = [
            [index for _, index in sorted(items)]
            for items in grouped.values()
            if {patch_idx for patch_idx, _ in items} == {1, 2, 3, 4}
        ]
        if not self.groups:
            raise ValueError("PanoramaBatchSampler found no complete four-patch panoramas.")

    def __iter__(self):
        order = np.arange(len(self.groups))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(order)
        self.epoch += 1

        batch = []
        for group_idx in order:
            batch.extend(self.groups[int(group_idx)])
            if len(batch) >= self.panoramas_per_batch * 4:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        full_batches = len(self.groups) // self.panoramas_per_batch
        if self.drop_last or len(self.groups) % self.panoramas_per_batch == 0:
            return full_batches
        return full_batches + 1


def dataloader_kwargs(num_workers: int) -> dict:
    """Use faster host-to-GPU transfer while keeping Windows worker settings explicit."""
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def smoke_test(config):
    """Run a smoke test to verify everything works."""
    print("\n" + "=" * 50)
    print("Running smoke test...")
    print("=" * 50)

    # Test dataset
    print("\n1. Testing dataset...")
    dataset_root = config['dataset']['root']
    report = create_dataset_report(dataset_root, "outputs/cache/smoke_dataset_report.json")
    print(f"Dataset report saved to outputs/cache/smoke_dataset_report.json")

    train_set = CVUSADataset(dataset_root, split="train", load_polar=True)
    val_set = CVUSADataset(dataset_root, split="val", load_polar=True)

    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")

    # Test loading
    sample = train_set[0]
    polar_shape = sample['polar'].shape if sample['polar'] is not None else 'N/A'
    print(f"Sample shapes:")
    print(f"  RGB: {sample['rgb'].shape}")
    print(f"  Gray: {sample['gray'].shape}")
    print(f"  Polar: {polar_shape}")
    print(f"  Satellite: {sample['satellite'].shape if sample['satellite'] is not None else 'N/A'}")

    # Test dataloader
    train_loader = DataLoader(
        train_set,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        **dataloader_kwargs(0),
    )

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Gray: {batch['gray'].shape}")
    print(f"  RGB: {batch['rgb'].shape}")
    print(f"  Polar: {batch['polar'].shape}")
    print(f"  Satellite: {batch['satellite'].shape}")
    print(f"  Patch idx: {batch['patch_idx'].shape}")

    print("\n" + "=" * 50)
    print("Smoke test passed!")
    print("=" * 50)


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command line args
    if args.resume:
        config['resume_from'] = args.resume
    if args.smoke_test:
        config['epochs'] = 2
        config['log_interval'] = 1
    if args.quick_train:
        config['training']['epochs'] = 1
        config['training']['max_train_batches'] = 2
        config['training']['max_val_batches'] = 2
        config['training']['log_interval'] = 1
        config['loss']['use_perceptual'] = False

    # Set seed
    set_seed(config.get('training', {}).get('seed', 42))
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    # Run smoke test first
    smoke_test(config)

    if args.smoke_test and not args.quick_train:
        print("\n" + "=" * 50)
        print("SMOKE TEST COMPLETE")
        print("=" * 50)
        return

    # Check DDColor weights
    ddcolor_weights = config.get('ddcolor', {}).get('weights_path')
    if not ddcolor_weights or not os.path.exists(ddcolor_weights):
        print(f"\nError: DDColor weights not found at {ddcolor_weights}")
        print("Please download ddcolor_paper_tiny weights and update config.yaml")
        return

    # Create dataloaders
    dataset_root = config['dataset']['root']
    semantic_cfg = config.get("semantic", {})
    correction_type = config["model"].get("correction_type")
    use_street_object_aware = correction_type in {
        "street_object_aware",
        "satellite_color_bottleneck",
        "satellite_detail_bottleneck_v15",
        "sky_gray_chroma_match_v20",
    }
    polar_size = tuple(config["model"].get("polar_input_size", [256, 512]))
    satellite_size = tuple(config["model"].get("satellite_input_size", [256, 256]))
    load_semantics = bool(
        (
            config["model"].get("use_semantic_cross_view_vit", False)
            or config["model"].get("use_semantic_color_token_match", False)
            or config["model"].get("use_polar_token_match", False)
            or config["model"].get("use_semantic_cnn_context", False)
            or use_street_object_aware
        )
        and semantic_cfg.get("load_precomputed", False)
    )
    train_set = CVUSADataset(
        dataset_root,
        split="train",
        load_polar=True,
        use_segmap=use_street_object_aware,
        load_semantics=load_semantics,
        street_semantic_dirname=semantic_cfg.get("street_dirname", "street_semantic"),
        satellite_semantic_dirname=semantic_cfg.get("satellite_dirname", "overhead_satellite_semantic"),
        polar_size=polar_size,
        satellite_size=satellite_size,
    )
    val_set = CVUSADataset(
        dataset_root,
        split="val",
        load_polar=True,
        use_segmap=use_street_object_aware,
        load_semantics=load_semantics,
        street_semantic_dirname=semantic_cfg.get("street_dirname", "street_semantic"),
        satellite_semantic_dirname=semantic_cfg.get("satellite_dirname", "overhead_satellite_semantic"),
        polar_size=polar_size,
        satellite_size=satellite_size,
    )

    panorama_cfg = config.get("panorama_consistency", {})
    use_panorama_batches = bool(
        panorama_cfg.get("enabled", False)
        or config["training"].get("use_panorama_batch_sampler", False)
    )
    if use_panorama_batches:
        train_batch_sampler = PanoramaBatchSampler(
            train_set,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            drop_last=True,
            seed=config["training"].get("seed", 42),
        )
        print(
            "[Train] Panorama batch sampler enabled: "
            f"{train_batch_sampler.panoramas_per_batch} panorama(s), "
            f"{train_batch_sampler.panoramas_per_batch * 4} patch(es) per batch."
        )
        train_loader = DataLoader(
            train_set,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            **dataloader_kwargs(config['training']['num_workers']),
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            **dataloader_kwargs(config['training']['num_workers']),
        )

    val_loader = DataLoader(
        val_set,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        **dataloader_kwargs(config['training']['num_workers']),
    )

    # Create model
    print("\n" + "=" * 50)
    print("Loading model...")
    print("=" * 50)

    try:
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
            gray_region_bins=config['model'].get('gray_region_bins', 8),
            chroma_region_bins=config['model'].get('chroma_region_bins', 8),
            gray_region_temperature=config['model'].get('gray_region_temperature', 0.035),
            chroma_region_temperature=config['model'].get('chroma_region_temperature', 0.045),
            street_semantic_source=config['model'].get('street_semantic_source', 'dino_v12'),
            satellite_semantic_source=config['model'].get('satellite_semantic_source', 'neos'),
            dino_model_name=config['model'].get('dino_model_name', 'vit_small_patch16_224.dino'),
            dino_pretrained=config['model'].get('dino_pretrained', True),
        )
    except Exception as e:
        print(f"\nError loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    init_from = config['training'].get('init_from')
    if init_from:
        print(f"\nWarm-starting matching weights from: {init_from}")
        checkpoint = torch.load(init_from, map_location=next(model.parameters()).device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        load_matching_state_dict(model, state_dict)

    # Create trainer
    training_config = {
        **config['training'],
        **config['loss'],
        'output_base_dir': config['paths']['output_base_dir'],
        'checkpoint_base_dir': config['paths']['checkpoint_base_dir'],
        'satellite_dependency': config.get('satellite_dependency', {}),
        'modulation_focus': config.get('modulation_focus', {}),
        'panorama_consistency': config.get('panorama_consistency', {}),
        'semantic_color_consistency': config.get('semantic_color_consistency', {}),
        'semantic_gate_prior': config.get('semantic_gate_prior', {}),
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        exp_name=args.exp_name,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    from pathlib import Path
    main()
