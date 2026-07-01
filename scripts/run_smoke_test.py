"""
Smoke test script to verify everything works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import cv2
import numpy as np

from src.dataset import CVUSADataset, create_dataset_report
from src.model import SatelliteGuidedDDColor
from src.utils import load_config


def test_dataset():
    """Test dataset loading."""
    print("\n" + "=" * 60)
    print("TEST 1: Dataset Loading")
    print("=" * 60)

    config = load_config("config.yaml")
    dataset_root = config['dataset']['root']

    print(f"Dataset root: {dataset_root}")

    # Create report
    report = create_dataset_report(dataset_root, "outputs/cache/dataset_report.json")
    print(f"Dataset report saved to outputs/cache/dataset_report.json")

    # Load train set
    train_set = CVUSADataset(dataset_root, split="train", load_polar=True)
    print(f"Train samples: {len(train_set)}")

    # Load a sample
    sample = train_set[0]
    polar_shape = sample['polar'].shape if sample['polar'] is not None else 'N/A'
    print(f"Sample shapes:")
    print(f"  RGB: {sample['rgb'].shape}")
    print(f"  Gray: {sample['gray'].shape}")
    print(f"  Polar: {polar_shape}")
    print(f"  Satellite: {sample['satellite'].shape if sample['satellite'] is not None else 'N/A'}")
    print(f"  File ID: {sample['file_id']}")

    print("\n" + "=" * 60)
    print("PASSED: Dataset loading works correctly")
    print("=" * 60)


def test_ddcolor():
    """Test DDColor loading."""
    print("\n" + "=" * 60)
    print("TEST 2: DDColor Loading")
    print("=" * 60)

    config = load_config("config.yaml")
    ddcolor_weights = config['ddcolor']['weights_path']

    print(f"DDColor weights: {ddcolor_weights}")

    if not Path(ddcolor_weights).exists():
        print(f"ERROR: DDColor weights not found!")
        print(f"Please download weights to: {ddcolor_weights}")
        return False

    try:
        model = SatelliteGuidedDDColor(
            ddcolor_weights_path=ddcolor_weights,
            ddcolor_code_path=config.get('ddcolor', {}).get('code_path'),
            correction_type="light",
            satellite_encoder_pretrained=config['model'].get('satellite_encoder_pretrained', True),
        )

        stats = model.get_num_parameters()
        print(f"DDColor total: {stats['ddcolor_total']:,}")
        print(f"DDColor frozen: {stats['ddcolor_frozen']:,}")
        print(f"Trainable: {stats['total_trainable']:,}")
        print(f"DDColor frozen verified: {model.verify_ddcolor_frozen()}")

        print("\n" + "=" * 60)
        print("PASSED: DDColor loaded and frozen correctly")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward():
    """Test forward pass."""
    print("\n" + "=" * 60)
    print("TEST 3: Forward Pass")
    print("=" * 60)

    config = load_config("config.yaml")

    try:
        model = SatelliteGuidedDDColor(
            ddcolor_weights_path=config['ddcolor']['weights_path'],
            ddcolor_code_path=config.get('ddcolor', {}).get('code_path'),
            correction_type="light",
            satellite_encoder_pretrained=config['model'].get('satellite_encoder_pretrained', True),
        )
        model.eval()

        device = next(model.parameters()).device
        print(f"Device: {device}")

        # Create dummy inputs
        gray_rgb = torch.randn(2, 3, 256, 256).to(device)
        polar_img = torch.randn(2, 3, 256, 512).to(device)
        satellite_img = torch.randn(2, 3, 256, 256).to(device)
        patch_idx = torch.tensor([1, 2], dtype=torch.long, device=device)

        print(
            f"Input shapes: gray={gray_rgb.shape}, polar={polar_img.shape}, "
            f"satellite={satellite_img.shape}, patch_idx={patch_idx.shape}"
        )

        with torch.no_grad():
            output = model(gray_rgb, polar_img, satellite_img, patch_idx)

        print(f"Output shapes:")
        print(f"  Base RGB: {output['base_rgb'].shape}")
        print(f"  Context: {output['context_vector'].shape}")
        print(f"  Final RGB: {output['final_rgb'].shape}")
        print(f"  Delta: {output['delta_color'].shape}")

        # Check output ranges
        print(f"Output ranges:")
        print(f"  Base RGB: [{output['base_rgb'].min():.4f}, {output['base_rgb'].max():.4f}]")
        print(f"  Final RGB: [{output['final_rgb'].min():.4f}, {output['final_rgb'].max():.4f}]")
        print(f"  Delta: [{output['delta_color'].min():.4f}, {output['delta_color'].max():.4f}]")

        print("\n" + "=" * 60)
        print("PASSED: Forward pass works correctly")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss():
    """Test loss computation."""
    print("\n" + "=" * 60)
    print("TEST 4: Loss Computation")
    print("=" * 60)

    from src.loss import TotalLoss

    try:
        criterion = TotalLoss(
            l1_weight=1.0,
            perceptual_weight=0.5,
            residual_weight=0.1,
            use_perceptual=True,
        )

        pred = torch.randn(2, 3, 256, 256)
        target = torch.randn(2, 3, 256, 256)
        delta = torch.randn(2, 3, 256, 256) * 0.1

        losses = criterion(pred, target, delta)

        print("Loss components:")
        for k, v in losses.items():
            print(f"  {k}: {v.item():.4f}")

        # Check for NaN
        for k, v in losses.items():
            if torch.isnan(v):
                print(f"ERROR: {k} loss is NaN!")
                return False

        print("\n" + "=" * 60)
        print("PASSED: Loss computation works correctly")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics computation."""
    print("\n" + "=" * 60)
    print("TEST 5: Metrics Computation")
    print("=" * 60)

    from src.metrics import MetricsCalculator

    try:
        calculator = MetricsCalculator()

        pred = torch.rand(4, 3, 256, 256)
        target = torch.rand(4, 3, 256, 256)

        metrics = calculator.compute_batch(pred, target)

        print("Metrics:")
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

        print("\n" + "=" * 60)
        print("PASSED: Metrics computation works correctly")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("SATELLITE-GUIDED DDCOLOR - SMOKE TEST")
    print("=" * 60)

    # Create output directory
    Path("outputs/cache").mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': test_dataset(),
        'ddcolor': test_ddcolor(),
        'forward': test_forward(),
        'loss': test_loss(),
        'metrics': test_metrics(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("=" * 60)

    if all_passed:
        print("All tests PASSED! You can now start training.")
        print("Run: python train.py --smoke_test")
    else:
        print("Some tests FAILED. Please check the errors above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
