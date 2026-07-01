"""
Dataset loader for CVUSA processed split dataset.

The dataset contains pre-processed 256x256 patches with the following structure:
- train/val/test/ground_rgb: RGB street view patches
- train/val/test/ground_gray: Grayscale street view patches
- train/val/test/overhead_polar: Polar coordinate satellite views
- train/val/test/overhead_satellite: Original overhead satellite views

File naming:
- street patches: {panorama_id}_{patch_index}.jpg
- shared polar context: {panorama_id}.png or {panorama_id}.jpg
- shared satellite context: {panorama_id}.png or {panorama_id}.jpg
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict

import cv2
import numpy as np
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class CVUSADataset(Dataset):
    """CVUSA dataset for satellite-guided street view colorization."""

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        use_segmap: bool = False,
        load_semantics: bool = False,
        street_semantic_dirname: str = "street_semantic",
        satellite_semantic_dirname: str = "overhead_satellite_semantic",
        load_polar: bool = True,
        load_satellite: bool = True,
        require_complete_panoramas: bool = True,
        polar_size: Tuple[int, int] = (256, 512),
        satellite_size: Tuple[int, int] = (256, 256),
    ):
        """
        Args:
            dataset_root: Path to CVUSA_processed_split directory
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply
            use_segmap: Whether to load segmentation maps
            load_polar: Whether to load polar satellite views
            load_satellite: Whether to load original overhead satellite views
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.use_segmap = use_segmap
        self.load_semantics = load_semantics
        self.street_semantic_dirname = street_semantic_dirname
        self.satellite_semantic_dirname = satellite_semantic_dirname
        self.load_polar = load_polar
        self.load_satellite = load_satellite
        self.require_complete_panoramas = require_complete_panoramas
        self.polar_size = polar_size
        self.satellite_size = satellite_size

        # Validate paths
        self._validate_paths()

        # Build index
        self.samples = self._build_index()

    def _validate_paths(self):
        """Validate that required directories exist."""
        required_dirs = [
            self.dataset_root / self.split / "ground_rgb",
            self.dataset_root / self.split / "ground_gray",
        ]
        if self.load_polar:
            required_dirs.append(self.dataset_root / self.split / "overhead_polar")
        if self.load_satellite:
            required_dirs.append(self.dataset_root / self.split / "overhead_satellite")
        if self.use_segmap:
            required_dirs.append(self.dataset_root / self.split / "overhead_polar_seg")
        if self.load_semantics:
            required_dirs.append(self.dataset_root / self.split / self.street_semantic_dirname)
            required_dirs.append(self.dataset_root / self.split / self.satellite_semantic_dirname)

        missing = [d for d in required_dirs if not d.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required directories: {[str(d) for d in missing]}"
            )

    def _build_index(self) -> List[Dict]:
        """Build index of samples with file paths.

        Actual naming conventions:
        - RGB: {panorama_id}_{patch_idx}.jpg
        - Gray: {panorama_id}_{patch_idx}_gray.jpg
        - Polar: {panorama_id}.png/.jpg (without patch index)
        - Satellite: {panorama_id}.png/.jpg (without patch index)
        """
        rgb_dir = self.dataset_root / self.split / "ground_rgb"
        gray_dir = self.dataset_root / self.split / "ground_gray"
        polar_dir = self.dataset_root / self.split / "overhead_polar" if self.load_polar else None
        satellite_dir = self.dataset_root / self.split / "overhead_satellite" if self.load_satellite else None
        seg_dir = self.dataset_root / self.split / "overhead_polar_seg" if self.use_segmap else None
        street_sem_dir = self.dataset_root / self.split / self.street_semantic_dirname if self.load_semantics else None
        sat_sem_dir = self.dataset_root / self.split / self.satellite_semantic_dirname if self.load_semantics else None

        samples = []
        missing_files = defaultdict(list)

        # Group files by panorama ID
        # RGB: {panorama_id}_{patch_idx}.jpg
        rgb_files = {f.stem: f for f in rgb_dir.glob("*.jpg") if f.is_file()}

        # Gray: {panorama_id}_{patch_idx}_gray.jpg
        # Extract {panorama_id}_{patch_idx} from the stem
        gray_files = {}
        for f in gray_dir.glob("*.jpg"):
            if f.is_file():
                stem = f.stem
                # Remove _gray suffix to get base ID
                if stem.endswith("_gray"):
                    base_id = stem[:-5]  # Remove "_gray"
                    gray_files[base_id] = f

        if self.load_polar:
            polar_files = self._build_shared_context_index(polar_dir)
        else:
            polar_files = {}

        if self.load_satellite:
            satellite_files = self._build_shared_context_index(satellite_dir)
        else:
            satellite_files = {}

        if self.use_segmap:
            seg_files = {f.stem: f for f in self._iter_image_files(seg_dir)}
        else:
            seg_files = {}

        if self.load_semantics:
            street_sem_files = {f.stem: f for f in self._iter_image_files(street_sem_dir)}
            sat_sem_files = self._build_shared_context_index(sat_sem_dir)
        else:
            street_sem_files = {}
            sat_sem_files = {}

        for file_id, rgb_path in rgb_files.items():
            # Extract panorama ID (remove patch index)
            parts = file_id.rsplit("_", 1)
            if len(parts) != 2:
                continue
            panorama_id, patch_idx = parts[0], parts[1]

            # Check corresponding files
            # Gray file has _gray suffix
            gray_path = gray_files.get(f"{panorama_id}_{patch_idx}")
            polar_path = polar_files.get(panorama_id)  # Polar is shared across patches
            satellite_path = satellite_files.get(panorama_id)  # Satellite is shared across patches

            if not gray_path:
                missing_files['gray'].append(file_id)
                continue

            if self.load_polar and not polar_path:
                missing_files['polar'].append(file_id)
                continue
            if self.load_satellite and not satellite_path:
                missing_files['satellite'].append(file_id)
                continue

            sample = {
                "file_id": file_id,
                "panorama_id": panorama_id,
                "patch_idx": int(patch_idx) if patch_idx.isdigit() else 0,
                "rgb_path": str(rgb_path),
                "gray_path": str(gray_path),
                "polar_path": str(polar_path) if polar_path else None,
                "satellite_path": str(satellite_path) if satellite_path else None,
            }

            if self.use_segmap:
                # Some CVUSA polar segmentation exports are patch-specific
                # ({panorama_id}_{patch_idx}.png), while others are shared.
                seg_path = seg_files.get(file_id) or seg_files.get(panorama_id)
                sample["seg_path"] = str(seg_path) if seg_path else None

            if self.load_semantics:
                street_sem_path = street_sem_files.get(file_id) or street_sem_files.get(f"{file_id}_semantic")
                sat_sem_path = sat_sem_files.get(panorama_id)
                if not street_sem_path:
                    missing_files["street_semantic"].append(file_id)
                    continue
                if not sat_sem_path:
                    missing_files["satellite_semantic"].append(file_id)
                    continue
                sample["street_semantic_path"] = str(street_sem_path)
                sample["satellite_semantic_path"] = str(sat_sem_path)

            samples.append(sample)

        if self.require_complete_panoramas:
            grouped = defaultdict(list)
            for sample in samples:
                grouped[sample["panorama_id"]].append(sample)
            samples = [
                sample
                for panorama_samples in grouped.values()
                if {s["patch_idx"] for s in panorama_samples} == {1, 2, 3, 4}
                for sample in panorama_samples
            ]

        samples.sort(key=lambda s: (s["panorama_id"], s["patch_idx"]))

        # Report missing files
        if any(missing_files.values()):
            print(f"[Dataset] Missing files in {self.split}:")
            for key, files in missing_files.items():
                if files:
                    print(f"  {key}: {len(files)} files")

        print(f"[Dataset] Loaded {len(samples)} samples from {self.split}")
        return samples

    def _iter_image_files(self, directory: Path):
        """Yield common image formats in a directory."""
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path

    def _build_shared_context_index(self, directory: Path) -> Dict[str, Path]:
        """Index shared context files by panorama ID."""
        files = {}
        for f in self._iter_image_files(directory):
            stem = f.stem
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                panorama_id = parts[0]
            else:
                panorama_id = stem
            if panorama_id not in files:
                files[panorama_id] = f
        return files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load images
        rgb = self._load_image(sample["rgb_path"])
        gray = self._load_image(sample["gray_path"], grayscale=True)

        polar = None
        if self.load_polar and sample["polar_path"]:
            polar = self._load_image(sample["polar_path"])
            polar = cv2.resize(
                polar, (self.polar_size[1], self.polar_size[0]), interpolation=cv2.INTER_LINEAR
            )

        satellite = None
        if self.load_satellite and sample["satellite_path"]:
            satellite = self._load_image(sample["satellite_path"])
            satellite = cv2.resize(
                satellite,
                (self.satellite_size[1], self.satellite_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        seg = None
        if self.use_segmap and sample.get("seg_path"):
            seg = self._load_segmap(sample["seg_path"])
            if seg is not None and seg.shape[:2] != self.polar_size:
                seg = cv2.resize(
                    seg,
                    (self.polar_size[1], self.polar_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

        street_semantic = None
        satellite_semantic = None
        if self.load_semantics:
            street_semantic = self._load_label_map(sample["street_semantic_path"], size=(256, 256))
            satellite_semantic = self._load_label_map(sample["satellite_semantic_path"], size=self.satellite_size)

        result = {
            "rgb": rgb,
            "gray": gray,
            "polar": polar,
            "satellite": satellite,
            "seg": seg,
            "street_semantic": street_semantic,
            "satellite_semantic": satellite_semantic,
            "file_id": sample["file_id"],
            "panorama_id": sample["panorama_id"],
            "patch_idx": sample["patch_idx"],
        }

        if self.transform:
            result = self.transform(result)

        return result

    def _load_image(self, path: str, grayscale: bool = False) -> np.ndarray:
        """Load image as numpy array, normalized to [0, 1]."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        if not grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (256, 256) and grayscale:
            raise ValueError(f"Expected 256x256 patch, got {img.shape[:2]}: {path}")
        if (
            img.shape[:2] != (256, 256)
            and "overhead_polar" not in path
            and "overhead_satellite" not in path
        ):
            raise ValueError(f"Expected 256x256 patch, got {img.shape[:2]}: {path}")
        return img.astype(np.float32) / 255.0

    def _load_segmap(self, path: str) -> np.ndarray:
        """Load segmentation map."""
        seg = cv2.imread(path, cv2.IMREAD_COLOR)
        if seg is None:
            return None
        return seg.astype(np.float32) / 255.0

    def _load_label_map(self, path: str, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """Load a single-channel semantic label map as int64 class IDs."""
        label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Failed to load semantic label map: {path}")
        if label.shape[:2] != size:
            label = cv2.resize(label, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        return label.astype(np.int64)

    def get_dataset_info(self) -> Dict:
        """Return dataset statistics."""
        return {
            "split": self.split,
            "num_samples": len(self),
            "num_panoramas": len(set(s["panorama_id"] for s in self.samples)),
            "patches_per_panorama": self._get_patches_per_panorama(),
        }

    def _get_patches_per_panorama(self) -> Dict:
        """Get distribution of patches per panorama."""
        counts = defaultdict(int)
        for s in self.samples:
            counts[s["panorama_id"]] += 1
        return dict(sorted(counts.items()))


def create_dataset_report(dataset_root: str, output_path: str = None):
    """Generate a detailed dataset inspection report."""
    report = {
        "dataset_root": str(dataset_root),
        "splits": {},
        "summary": {}
    }

    for split in ["train", "val", "test"]:
        try:
            dataset = CVUSADataset(dataset_root, split=split, load_polar=True, load_satellite=True)
            info = dataset.get_dataset_info()
            report["splits"][split] = info
        except Exception as e:
            report["splits"][split] = {"error": str(e)}

    # Summary
    total_samples = sum(s.get("num_samples", 0) for s in report["splits"].values())
    report["summary"] = {
        "total_samples": total_samples,
        "splits_with_errors": [k for k, v in report["splits"].items() if "error" in v]
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    # Test dataset
    dataset_root = r"C:\Users\31133\Desktop\dataset1\CVUSA_processed_split"
    report = create_dataset_report(dataset_root, "outputs/cache/dataset_report.json")
    print(json.dumps(report, indent=2))

    # Test loading a sample
    dataset = CVUSADataset(dataset_root, split="train")
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  RGB: {sample['rgb'].shape}")
    print(f"  Gray: {sample['gray'].shape}")
    print(f"  Polar: {sample['polar'].shape if sample['polar'] is not None else 'N/A'}")
    print(f"  Satellite: {sample['satellite'].shape if sample['satellite'] is not None else 'N/A'}")
