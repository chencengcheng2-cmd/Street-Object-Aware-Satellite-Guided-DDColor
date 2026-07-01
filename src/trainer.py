"""
Training loop and utilities.
"""

import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from contextlib import nullcontext

from .loss import TotalLoss
from .metrics import MetricsCalculator
from .utils import (
    save_checkpoint, load_checkpoint, count_trainable_parameters,
    AverageMeter, set_seed, get_output_dir, get_checkpoint_dir, get_device,
)


class Trainer:
    """Trainer for Satellite-Guided DDColor."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        exp_name: str = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Setup directories
        if exp_name:
            self.exp_name = exp_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_name = f"exp_{timestamp}"

        self.output_dir = get_output_dir(self.exp_name, config.get('output_base_dir', 'outputs'))
        self.checkpoint_dir = get_checkpoint_dir(self.exp_name, config.get('checkpoint_base_dir', 'checkpoints'))

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir / "visualizations", exist_ok=True)
        os.makedirs(self.output_dir / "logs", exist_ok=True)

        # Setup device
        self.device = get_device()
        print(f"Using device: {self.device}")

        # Setup loss
        self.criterion = TotalLoss(
            l1_weight=config.get('l1_weight', 1.0),
            perceptual_weight=config.get('perceptual_weight', 0.5),
            residual_weight=config.get('residual_weight', 0.1),
            lane_marking_weight=config.get('lane_marking_weight', 0.0),
            edge_weight=config.get('edge_weight', 0.0),
            chroma_weight=config.get('chroma_weight', 0.0),
            colorful_weight=config.get('colorful_weight', 0.0),
            edge_chroma_weight=config.get('edge_chroma_weight', 0.0),
            fine_detail_color_weight=config.get('fine_detail_color_weight', 0.0),
            base_improvement_weight=config.get('base_improvement_weight', 0.0),
            use_perceptual=config.get('use_perceptual', True),
        ).to(self.device)

        # Setup optimizer (only trainable parameters)
        trainable_params = model.get_trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4),
        )

        # Verify DDColor is not in optimizer
        ddcolor_params = set(id(p) for p in model.ddcolor.parameters())
        optimizer_params = set(id(p) for group in self.optimizer.param_groups for p in group['params'])
        if ddcolor_params & optimizer_params:
            raise RuntimeError("Optimizer contains DDColor parameters! DDColor must be frozen.")

        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 30),
            eta_min=config.get('min_lr', 1e-6),
        )

        # Mixed precision
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Metrics
        self.metrics_calc = MetricsCalculator(
            device=str(self.device),
            enable_lpips=config.get('validate_lpips', False),
            enable_fid=config.get('validate_fid', False),
        )

        # Training state
        self.current_epoch = 0
        self.best_psnr = 0
        self.global_step = 0
        self.start_epoch = 0
        batch_metrics_cfg = config.get("batch_metrics", {})
        self.batch_metrics_enabled = bool(batch_metrics_cfg.get("enabled", False))
        self.batch_metrics_interval = int(batch_metrics_cfg.get("interval", config.get("log_interval", 100)))
        self.batch_metrics_interval = max(self.batch_metrics_interval, 1)
        if self.batch_metrics_enabled:
            print(
                "[Trainer] Batch metrics enabled: "
                f"interval={self.batch_metrics_interval}, "
                "metrics=(base_psnr, final_psnr, delta_psnr, final_l1, gate_mean)"
            )

        # Load checkpoint if specified
        if config.get('resume_from'):
            self.load_checkpoint(config['resume_from'])

        # Set seed
        set_seed(config.get('seed', 42))
        dep_cfg = config.get("satellite_dependency", {})
        self.satellite_dependency_enabled = bool(dep_cfg.get("enabled", False))
        self.satellite_dependency_weight = float(dep_cfg.get("weight", 0.0))
        self.satellite_dependency_margin = float(dep_cfg.get("margin", 0.02))
        self.satellite_dependency_warmup_epochs = int(dep_cfg.get("warmup_epochs", 0))
        self.satellite_dependency_mask_sky = bool(dep_cfg.get("mask_sky", True))
        self.satellite_dependency_negative_modes = tuple(dep_cfg.get("negative_modes", ["wrong"]))
        if self.satellite_dependency_enabled:
            print(
                "[Trainer] Satellite dependency loss enabled: "
                f"weight={self.satellite_dependency_weight}, "
                f"margin={self.satellite_dependency_margin}, "
                f"warmup_epochs={self.satellite_dependency_warmup_epochs}, "
                f"mask_sky={self.satellite_dependency_mask_sky}, "
                f"negative_modes={self.satellite_dependency_negative_modes}"
            )

        mod_cfg = config.get("modulation_focus", {})
        self.modulation_focus_enabled = bool(mod_cfg.get("enabled", False))
        self.modulation_focus_weight = float(mod_cfg.get("weight", 0.0))
        self.modulation_focus_strength = float(mod_cfg.get("strength", 2.0))
        self.modulation_gate_weight = float(mod_cfg.get("gate_weight", 0.0))
        self.modulation_preserve_weight = float(mod_cfg.get("preserve_weight", 0.0))
        self.modulation_threshold = float(mod_cfg.get("threshold", 0.06))
        self.modulation_temperature = float(mod_cfg.get("temperature", 0.04))
        if self.modulation_focus_enabled:
            print(
                "[Trainer] Error-aware modulation enabled: "
                f"weight={self.modulation_focus_weight}, "
                f"strength={self.modulation_focus_strength}, "
                f"gate_weight={self.modulation_gate_weight}, "
                f"preserve_weight={self.modulation_preserve_weight}, "
                f"threshold={self.modulation_threshold}, "
                f"temperature={self.modulation_temperature}"
            )

        pano_cfg = config.get("panorama_consistency", {})
        self.panorama_consistency_enabled = bool(pano_cfg.get("enabled", False))
        self.panorama_consistency_weight = float(pano_cfg.get("weight", 0.0))
        self.panorama_consistency_classes = tuple(pano_cfg.get("classes", [1, 2, 3, 4]))
        self.panorama_consistency_min_pixels = int(pano_cfg.get("min_pixels", 256))
        if self.panorama_consistency_enabled:
            print(
                "[Trainer] Panorama semantic color consistency enabled: "
                f"weight={self.panorama_consistency_weight}, "
                f"classes={self.panorama_consistency_classes}, "
                f"min_pixels={self.panorama_consistency_min_pixels}"
            )

        sem_color_cfg = config.get("semantic_color_consistency", {})
        self.semantic_color_consistency_enabled = bool(sem_color_cfg.get("enabled", False))
        self.semantic_color_consistency_weight = float(sem_color_cfg.get("weight", 0.0))
        self.semantic_color_consistency_class_weights = {
            int(k): float(v) for k, v in sem_color_cfg.get(
                "class_weights",
                {1: 0.7, 2: 0.3, 3: 1.0, 4: 1.0, 6: 0.2},
            ).items()
        }
        self.semantic_color_consistency_min_pixels = int(sem_color_cfg.get("min_pixels", 128))
        if self.semantic_color_consistency_enabled:
            print(
                "[Trainer] Semantic color consistency enabled: "
                f"weight={self.semantic_color_consistency_weight}, "
                f"class_weights={self.semantic_color_consistency_class_weights}, "
                f"min_pixels={self.semantic_color_consistency_min_pixels}"
            )

        gate_cfg = config.get("semantic_gate_prior", {})
        self.semantic_gate_prior_enabled = bool(gate_cfg.get("enabled", False))
        self.semantic_gate_prior_weight = float(gate_cfg.get("weight", 0.0))
        self.semantic_gate_prior_targets = {
            int(k): float(v) for k, v in gate_cfg.get(
                "targets",
                {0: 0.0, 1: 0.75, 2: 0.35, 3: 0.85, 4: 0.85, 5: 0.0, 6: 0.2},
            ).items()
        }
        self.semantic_gate_prior_min_pixels = int(gate_cfg.get("min_pixels", 128))
        if self.semantic_gate_prior_enabled:
            print(
                "[Trainer] Semantic gate prior enabled: "
                f"weight={self.semantic_gate_prior_weight}, "
                f"targets={self.semantic_gate_prior_targets}, "
                f"min_pixels={self.semantic_gate_prior_min_pixels}"
            )

        # Print model info
        self._print_model_info()

    def _print_model_info(self):
        """Print model parameter information."""
        stats = self.model.get_num_parameters()

        print("=" * 50)
        print("Model Parameters")
        print("=" * 50)
        print(f"DDColor total:      {stats['ddcolor_total']:,}")
        print(f"DDColor frozen:     {stats['ddcolor_frozen']:,}")
        print(f"Trainable:          {stats['total_trainable']:,}")
        print(f"Frozen:             {stats['total_frozen']:,}")
        print(f"Total:              {stats['total']:,}")
        print(f"DDColor frozen:     {self.model.verify_ddcolor_frozen()}")
        print("=" * 50)

        print("\nTrainable modules:")
        for name, num in stats['trainable_params']:
            print(f"  {name}: {num:,}")

        print("=" * 50)

    def train_epoch(self) -> Dict:
        """Train for one epoch."""
        self.model.train()
        losses = AverageMeter()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            gray_rgb = batch['gray'].to(self.device)
            rgb_gt = batch['rgb'].to(self.device)
            polar = batch['polar'].to(self.device)
            polar_seg = batch.get('polar_seg')
            if polar_seg is not None:
                polar_seg = polar_seg.to(self.device)
            satellite = batch['satellite'].to(self.device)
            patch_idx = batch['patch_idx'].to(self.device)
            street_semantic = batch.get('street_semantic')
            satellite_semantic = batch.get('satellite_semantic')
            if street_semantic is not None:
                street_semantic = street_semantic.to(self.device)
            if satellite_semantic is not None:
                satellite_semantic = satellite_semantic.to(self.device)

            # Forward pass
            amp_context = torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()
            with amp_context:
                output = self.model(
                    gray_rgb,
                    polar,
                    satellite,
                    patch_idx,
                    polar_seg=polar_seg,
                    street_semantic=street_semantic,
                    satellite_semantic=satellite_semantic,
                )
                loss_dict = self.criterion(
                    output['final_rgb'],
                    rgb_gt,
                    output['delta_color'],
                    base=output.get('base_rgb'),
                )
                loss = loss_dict['total']
                if self.modulation_focus_enabled and output.get("base_rgb") is not None:
                    focus_loss_dict = self._error_aware_modulation_loss(output, rgb_gt)
                    for key, value in focus_loss_dict.items():
                        loss_dict[key] = value
                    loss = loss + focus_loss_dict["modulation_focus_total"]

                if (
                    self.panorama_consistency_enabled
                    and self.panorama_consistency_weight > 0
                    and batch.get("panorama_id") is not None
                ):
                    pano_loss = self._panorama_semantic_color_consistency_loss(output, batch)
                    loss_dict["panorama_consistency"] = pano_loss
                    loss = loss + self.panorama_consistency_weight * pano_loss

                if (
                    self.semantic_color_consistency_enabled
                    and self.semantic_color_consistency_weight > 0
                ):
                    sem_color_loss = self._semantic_color_consistency_loss(output)
                    loss_dict["semantic_color_consistency"] = sem_color_loss
                    loss = loss + self.semantic_color_consistency_weight * sem_color_loss

                if (
                    self.semantic_gate_prior_enabled
                    and self.semantic_gate_prior_weight > 0
                ):
                    gate_prior_loss = self._semantic_gate_prior_loss(output)
                    loss_dict["semantic_gate_prior"] = gate_prior_loss
                    loss = loss + self.semantic_gate_prior_weight * gate_prior_loss

                if (
                    self.satellite_dependency_enabled
                    and self.satellite_dependency_weight > 0
                    and gray_rgb.size(0) > 1
                    and self.current_epoch >= self.satellite_dependency_warmup_epochs
                ):
                    dep_mask = None
                    if self.satellite_dependency_mask_sky:
                        dep_mask = output.get("non_sky_mask")
                        if dep_mask is not None:
                            dep_mask = dep_mask.detach().to(device=rgb_gt.device, dtype=rgb_gt.dtype)

                    correct_l1 = self._masked_sample_l1(output["final_rgb"], rgb_gt, dep_mask)
                    dependency_terms = []
                    for negative_mode in self.satellite_dependency_negative_modes:
                        neg_satellite, neg_satellite_semantic, neg_polar, neg_polar_seg = self._make_satellite_negative(
                            satellite,
                            satellite_semantic,
                            polar,
                            polar_seg,
                            negative_mode,
                        )
                        if neg_satellite is None:
                            continue
                        negative_output = self.model(
                            gray_rgb,
                            neg_polar,
                            neg_satellite,
                            patch_idx,
                            polar_seg=neg_polar_seg,
                            street_semantic=street_semantic,
                            satellite_semantic=neg_satellite_semantic,
                        )
                        negative_l1 = self._masked_sample_l1(negative_output["final_rgb"], rgb_gt, dep_mask)
                        dependency_terms.append(
                            torch.relu(correct_l1 - negative_l1 + self.satellite_dependency_margin).mean()
                        )
                    dependency_loss = (
                        torch.stack(dependency_terms).mean()
                        if dependency_terms
                        else output["final_rgb"].new_tensor(0.0)
                    )
                    loss_dict["satellite_dependency"] = dependency_loss
                    loss = loss + self.satellite_dependency_weight * dependency_loss

            # Check for NaN
            if not torch.isfinite(loss):
                print(f"NaN loss at step {self.global_step}!")
                print(f"Loss dict: {loss_dict}")
                continue

            # Backward pass
            self.optimizer.zero_grad()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            losses.update(loss.item(), gray_rgb.size(0))

            # Log progress
            if batch_idx % self.config.get('log_interval', 100) == 0:
                metric_text = ""
                if self.batch_metrics_enabled and batch_idx % self.batch_metrics_interval == 0:
                    metric_text = " " + self._format_batch_metrics(output, rgb_gt)
                print(
                    f"Epoch [{self.current_epoch}/{self.config['epochs']}][{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {losses.avg:.4f} "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                    f"{metric_text}"
                )

            self.global_step += 1
            if self.config.get('max_train_batches') and batch_idx + 1 >= self.config['max_train_batches']:
                break

        return {'total_loss': losses.avg}

    @staticmethod
    def _batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred.detach().float(), target.detach().float())
        return -10.0 * torch.log10(mse.clamp_min(1e-10))

    def _format_batch_metrics(self, output: Dict, rgb_gt: torch.Tensor) -> str:
        with torch.no_grad():
            final = output["final_rgb"].detach()
            base = output.get("base_rgb")
            final_psnr = self._batch_psnr(final, rgb_gt).item()
            final_l1 = F.l1_loss(final.float(), rgb_gt.float()).item()
            if base is not None:
                base_psnr = self._batch_psnr(base.detach(), rgb_gt).item()
                delta_psnr = final_psnr - base_psnr
            else:
                base_psnr = float("nan")
                delta_psnr = float("nan")
            gate = output.get("satellite_gate")
            gate_mean = gate.detach().float().mean().item() if gate is not None else float("nan")
        return (
            f"base_psnr={base_psnr:.2f} "
            f"final_psnr={final_psnr:.2f} "
            f"delta_psnr={delta_psnr:+.2f} "
            f"final_l1={final_l1:.4f} "
            f"gate_mean={gate_mean:.3f}"
        )

    @staticmethod
    def _masked_sample_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Per-sample L1, optionally restricted to a spatial mask."""
        if mask is not None:
            mask = mask.to(device=pred.device, dtype=pred.dtype)
            denom = mask.flatten(1).sum(dim=1).clamp_min(1.0) * pred.shape[1]
            return ((pred - target).abs() * mask).flatten(1).sum(dim=1) / denom
        return (pred - target).abs().flatten(1).mean(dim=1)

    @staticmethod
    def _make_satellite_negative(
        satellite: Optional[torch.Tensor],
        satellite_semantic: Optional[torch.Tensor],
        polar: Optional[torch.Tensor],
        polar_seg: Optional[torch.Tensor],
        mode: str,
    ):
        """Create a satellite negative sample for dependency training."""
        if satellite is None:
            return None, satellite_semantic, polar, polar_seg

        mode = str(mode).lower()
        if mode in {"wrong", "rolled", "mismatch"}:
            neg_satellite = torch.roll(satellite, shifts=1, dims=0)
            neg_semantic = torch.roll(satellite_semantic, shifts=1, dims=0) if satellite_semantic is not None else satellite_semantic
            neg_polar = torch.roll(polar, shifts=1, dims=0) if polar is not None else polar
            neg_polar_seg = torch.roll(polar_seg, shifts=1, dims=0) if polar_seg is not None else polar_seg
        elif mode in {"gray", "grey", "grayscale"}:
            gray = satellite.mean(dim=1, keepdim=True)
            neg_satellite = gray.repeat(1, satellite.shape[1], 1, 1)
            neg_semantic = satellite_semantic
            neg_polar = polar
            neg_polar_seg = polar_seg
        elif mode in {"channel_shuffle", "color_shuffle", "shuffled"}:
            if satellite.shape[1] >= 3:
                neg_satellite = satellite[:, [2, 0, 1], :, :]
            else:
                neg_satellite = satellite.flip(1)
            neg_semantic = satellite_semantic
            neg_polar = polar
            neg_polar_seg = polar_seg
        else:
            raise ValueError(f"Unknown satellite dependency negative mode: {mode}")

        return neg_satellite, neg_semantic, neg_polar, neg_polar_seg

    def _semantic_color_consistency_loss(self, output: Dict) -> torch.Tensor:
        """Force non-street-only regions to stay close to satellite color prototypes.

        This is the direct counterpart of a geometry consistency loss: the model
        is not allowed to ignore the injected satellite semantic color prior.
        Chroma is used so luminance/shadows can still come from DDColor/street.
        """
        pred = output.get("final_rgb")
        prior = output.get("satellite_color_prior")
        semantic = output.get("semantic_street_map")
        if pred is None or prior is None or semantic is None:
            ref = pred if pred is not None else prior
            return ref.new_tensor(0.0) if ref is not None else torch.tensor(0.0, device=self.device)

        if prior.shape[-2:] != pred.shape[-2:]:
            prior = F.interpolate(prior, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        if semantic.shape[-2:] != pred.shape[-2:]:
            semantic = F.interpolate(semantic.float(), size=pred.shape[-2:], mode="nearest")

        labels = semantic.detach().argmax(dim=1, keepdim=True)
        pred_chroma = pred - pred.mean(dim=1, keepdim=True)
        prior_chroma = prior.detach() - prior.detach().mean(dim=1, keepdim=True)

        losses = []
        for class_id, weight in self.semantic_color_consistency_class_weights.items():
            if weight <= 0:
                continue
            mask = (labels == int(class_id)).to(dtype=pred.dtype)
            if mask.sum() < self.semantic_color_consistency_min_pixels:
                continue
            denom = mask.sum().clamp_min(1.0) * pred.shape[1]
            class_loss = ((pred_chroma - prior_chroma).abs() * mask).sum() / denom
            losses.append(float(weight) * class_loss)

        if not losses:
            return pred.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _semantic_gate_prior_loss(self, output: Dict) -> torch.Tensor:
        """Regularize the satellite gate with semantic reliability priors."""
        gate = output.get("satellite_gate")
        semantic = output.get("semantic_street_map")
        if gate is None or semantic is None:
            ref = gate if gate is not None else output.get("final_rgb")
            return ref.new_tensor(0.0) if ref is not None else torch.tensor(0.0, device=self.device)

        if semantic.shape[-2:] != gate.shape[-2:]:
            semantic = F.interpolate(semantic.float(), size=gate.shape[-2:], mode="nearest")
        labels = semantic.detach().argmax(dim=1, keepdim=True)

        losses = []
        for class_id, target_value in self.semantic_gate_prior_targets.items():
            mask = (labels == int(class_id)).to(device=gate.device, dtype=gate.dtype)
            if mask.sum() < self.semantic_gate_prior_min_pixels:
                continue
            target = torch.full_like(gate, float(target_value))
            losses.append((F.mse_loss(gate * mask, target * mask, reduction="sum") / mask.sum().clamp_min(1.0)))

        if not losses:
            return gate.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _error_aware_modulation_loss(self, output: Dict, rgb_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Supervise where modulation should be strong or weak.

        DDColor-vs-GT error is available only during training. We use it to
        teach the gate:
        - high base error -> stronger correction pressure
        - low base error -> preserve DDColor and keep gate/delta small
        """
        pred = output["final_rgb"]
        base = output["base_rgb"].detach()
        delta = output["delta_color"]

        base_err = (base - rgb_gt).abs().mean(dim=1, keepdim=True)
        target_gate = torch.sigmoid(
            (base_err - self.modulation_threshold) / max(self.modulation_temperature, 1e-6)
        ).detach()

        mask = output.get("non_sky_mask")
        if mask is None:
            mask = torch.ones_like(target_gate)
        else:
            mask = mask.detach().to(device=rgb_gt.device, dtype=rgb_gt.dtype)

        denom = mask.sum().clamp_min(1.0)
        weighted_l1 = (
            (pred - rgb_gt).abs().mean(dim=1, keepdim=True)
            * (1.0 + self.modulation_focus_strength * target_gate)
            * mask
        ).sum() / denom

        preserve_target = (1.0 - target_gate) * mask
        preserve_loss = (delta.abs().mean(dim=1, keepdim=True) * preserve_target).sum() / preserve_target.sum().clamp_min(1.0)

        gate_loss = pred.new_tensor(0.0)
        gate = output.get("satellite_gate")
        if gate is not None:
            gate = gate.to(device=rgb_gt.device, dtype=rgb_gt.dtype)
            gate_loss = F.mse_loss(gate * mask, target_gate * mask, reduction="sum") / denom

        total = (
            self.modulation_focus_weight * weighted_l1
            + self.modulation_gate_weight * gate_loss
            + self.modulation_preserve_weight * preserve_loss
        )
        return {
            "modulation_focus_l1": weighted_l1,
            "modulation_gate": gate_loss,
            "modulation_preserve": preserve_loss,
            "modulation_focus_total": total,
        }

    def _panorama_semantic_color_consistency_loss(
        self,
        output: Dict,
        batch: Dict,
    ) -> torch.Tensor:
        """Keep same-semantic chroma consistent across patches of a panorama.

        This is the train-time counterpart of inference-time Lab harmonization.
        It is differentiable and only uses chroma, so it does not flatten
        brightness, shadows, or DDColor's luminance structure.
        """
        pred = output["final_rgb"]
        semantic = output.get("semantic_street_map")
        panorama_ids = batch.get("panorama_id")
        if semantic is None or panorama_ids is None:
            return pred.new_tensor(0.0)

        if semantic.shape[-2:] != pred.shape[-2:]:
            semantic = F.interpolate(semantic.float(), size=pred.shape[-2:], mode="nearest")
        labels = semantic.detach().argmax(dim=1)
        chroma = pred - pred.mean(dim=1, keepdim=True)

        losses = []
        unique_ids = sorted(set(panorama_ids))
        for pano_id in unique_ids:
            indices = [idx for idx, item in enumerate(panorama_ids) if item == pano_id]
            if len(indices) < 2:
                continue
            for class_id in self.panorama_consistency_classes:
                means = []
                for idx in indices:
                    mask = (labels[idx:idx + 1] == int(class_id)).float().unsqueeze(1)
                    pixel_count = mask.sum()
                    if pixel_count < self.panorama_consistency_min_pixels:
                        continue
                    mean_chroma = (chroma[idx:idx + 1] * mask).sum(dim=(2, 3)) / pixel_count.clamp_min(1.0)
                    means.append(mean_chroma.squeeze(0))
                if len(means) < 2:
                    continue
                class_means = torch.stack(means, dim=0)
                target = class_means.mean(dim=0, keepdim=True)
                losses.append((class_means - target).abs().mean())

        if not losses:
            return pred.new_tensor(0.0)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """Validate the model."""
        self.model.eval()

        psnr_values = []
        ssim_values = []
        lpips_values = []

        if self.metrics_calc.fid:
            self.metrics_calc.fid.reset()

        for batch_idx, batch in enumerate(self.val_loader):
            gray_rgb = batch['gray'].to(self.device)
            rgb_gt = batch['rgb'].to(self.device)
            polar = batch['polar'].to(self.device)
            polar_seg = batch.get('polar_seg')
            if polar_seg is not None:
                polar_seg = polar_seg.to(self.device)
            satellite = batch['satellite'].to(self.device)
            patch_idx = batch['patch_idx'].to(self.device)
            street_semantic = batch.get('street_semantic')
            satellite_semantic = batch.get('satellite_semantic')
            if street_semantic is not None:
                street_semantic = street_semantic.to(self.device)
            if satellite_semantic is not None:
                satellite_semantic = satellite_semantic.to(self.device)

            # Forward
            amp_context = torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()
            with amp_context:
                output = self.model(
                    gray_rgb,
                    polar,
                    satellite,
                    patch_idx,
                    polar_seg=polar_seg,
                    street_semantic=street_semantic,
                    satellite_semantic=satellite_semantic,
                )

            # Compute metrics
            metrics = self.metrics_calc.compute_batch(output['final_rgb'], rgb_gt)

            if metrics['psnr'] != float('inf'):
                psnr_values.append(metrics['psnr'])
            ssim_values.append(metrics['ssim'])
            if metrics['lpips'] is not None:
                lpips_values.append(metrics['lpips'])
            if self.config.get('max_val_batches') and batch_idx + 1 >= self.config['max_val_batches']:
                break

        # Compute average metrics
        results = {
            'psnr': np.mean(psnr_values) if psnr_values else 0,
            'ssim': np.mean(ssim_values) if ssim_values else 0,
            'lpips': np.mean(lpips_values) if lpips_values else None,
        }

        # Get FID
        fid_val = self.metrics_calc.get_fid()
        if fid_val is not None:
            results['fid'] = fid_val

        print(f"Validation - PSNR: {results['psnr']:.2f}, SSIM: {results['ssim']:.4f}")
        if results.get('lpips') is not None:
            print(f"Validation - LPIPS: {results['lpips']:.4f}")
        if results.get('fid') is not None:
            print(f"Validation - FID: {results['fid']:.2f}")

        return results

    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{self.current_epoch}.pth"

        metrics = {
            'best_psnr': self.best_psnr,
            'global_step': self.global_step,
        }

        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.current_epoch,
            metrics,
            str(checkpoint_path),
            is_best=is_best,
        )

        # Also save latest
        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.current_epoch,
            metrics,
            str(self.checkpoint_dir / "latest.pth"),
            is_best=False,
        )

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        epoch, metrics = load_checkpoint(path, self.model, self.optimizer, self.scheduler)
        self.start_epoch = epoch + 1
        self.current_epoch = epoch
        self.best_psnr = metrics.get('best_psnr', 0)
        self.global_step = metrics.get('global_step', 0)
        print(f"Resumed from epoch {epoch}, best PSNR: {self.best_psnr:.2f}")

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config['epochs']} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        print(f"Outputs will be saved to: {self.output_dir}")
        print()

        start_time = time.time()

        for epoch in range(self.start_epoch, self.config['epochs']):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate(epoch)

                # Update best PSNR
                if val_metrics['psnr'] > self.best_psnr:
                    self.best_psnr = val_metrics['psnr']
                    print(f"New best PSNR: {self.best_psnr:.2f}")
                    is_best = True
                else:
                    is_best = False

                # Save checkpoint
                if (epoch + 1) % self.config.get('save_interval', 1) == 0:
                    self.save_checkpoint(is_best=is_best)

            # Update learning rate
            self.scheduler.step()

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time / 3600:.2f} hours")
        print(f"Best PSNR: {self.best_psnr:.2f}")
