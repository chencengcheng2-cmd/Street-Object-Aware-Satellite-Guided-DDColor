"""
Main model combining Frozen DDColor, satellite/polar context encoders, FiLM and residual correction.
"""

import torch
import torch.nn as nn

from .cross_view_vit import CrossViewViTEncoder
from .semantic_cross_view_vit import SemanticGuidedCrossViewViTEncoder
from .semantic_color_token_matcher import SemanticColorTokenMatcher
from .semantic_context import SemanticSatelliteStatsEncoder
from .ddcolor_wrapper import DDColorWrapper
from .polar_encoder import PolarContextEncoder
from .correction_module import (
    DetailAwareLightCorrectionModule,
    LightCorrectionModule,
    SemanticAwareCorrectionModule,
    TokenColorCorrectionModule,
    ResidualCorrectionModule,
)
from .lane_vit_encoder import LaneViTDetailEncoder
from .satellite_vit_encoder import SatelliteViTColorEncoder
from .street_object_aware import StreetObjectAwareCorrectionModule
from .satellite_color_bottleneck import SatelliteColorBottleneckCorrectionModule
from .satellite_detail_bottleneck import SatelliteDetailBottleneckCorrectionModule
from .sky_gray_chroma_match import SkyOnlyGrayChromaMatchCorrectionModule


class SatelliteGuidedDDColor(nn.Module):
    """
    Satellite-guided DDColor Enhancement Model.

    Architecture:
    1. Frozen DDColor provides base colorization
    2. Satellite Encoder extracts global overhead context
    3. Polar Encoder extracts directional/detail context
    4. Residual Correction Module (with optional FiLM) applies corrections
    """

    def __init__(
        self,
        ddcolor_weights_path: str,
        ddcolor_code_path: str = None,
        context_dim: int = 512,
        polar_encoder_pretrained: bool = True,
        satellite_encoder_pretrained: bool = True,
        correction_type: str = "resnet",  # 'resnet' or 'light'
        residual_scale: float = 0.1,
        polar_input_size: tuple = (256, 512),
        use_film: bool = True,
        use_polar_context: bool = True,
        use_lane_vit: bool = False,
        lane_vit_embed_dim: int = 192,
        lane_vit_depth: int = 4,
        lane_vit_heads: int = 3,
        lane_vit_patch_size: int = 16,
        lane_feature_dim: int = 64,
        use_satellite_vit: bool = False,
        satellite_vit_embed_dim: int = 192,
        satellite_vit_depth: int = 4,
        satellite_vit_heads: int = 3,
        satellite_vit_patch_size: int = 16,
        satellite_vit_feature_dim: int = 64,
        use_cross_view_vit: bool = False,
        use_semantic_cross_view_vit: bool = False,
        use_semantic_color_token_match: bool = False,
        use_polar_token_match: bool = False,
        use_semantic_cnn_context: bool = False,
        semantic_num_classes: int = 6,
        cross_view_embed_dim: int = 192,
        cross_view_depth: int = 3,
        cross_view_heads: int = 3,
        cross_view_patch_size: int = 16,
        cross_view_street_patch_size: int = 16,
        cross_view_satellite_patch_size: int = 8,
        cross_view_feature_dim: int = 64,
        color_token_match_weight: float = 3.0,
        semantic_distribution_weight: float = 2.0,
        boundary_match_weight: float = 2.0,
        token_delta_scale: float = 0.35,
        token_correction_scale: float = 0.8,
        street_object_hidden_dim: int = 96,
        street_object_num_masks: int = 8,
        street_object_detail_scale: float = 0.18,
        satellite_prior_strength: float = 0.65,
        use_street_gray_edges: bool = False,
        use_street_gray_modulation: bool = True,
        use_gray_satellite_token_selection: bool = True,
        use_satellite_chroma_token_selection: bool = False,
        token_selection_patch_size: int = 16,
        token_selection_dim: int = 32,
        lane_detail_strength: float = 0.45,
        satellite_dependency_boost: float = 1.35,
        lane_evidence_threshold: float = 0.002,
        gray_region_bins: int = 8,
        chroma_region_bins: int = 8,
        gray_region_temperature: float = 0.035,
        chroma_region_temperature: float = 0.045,
        street_semantic_source: str = "dino_v12",
        satellite_semantic_source: str = "neos",
        dino_model_name: str = "vit_small_patch16_224.dino",
        dino_pretrained: bool = True,
        device: str = "auto",
    ):
        super().__init__()
        self.context_dim = context_dim
        self.residual_scale = residual_scale
        self.polar_input_size = polar_input_size
        self.use_polar_context = use_polar_context
        self.use_lane_vit = use_lane_vit
        self.lane_feature_dim = lane_feature_dim
        self.use_satellite_vit = use_satellite_vit
        self.satellite_vit_feature_dim = satellite_vit_feature_dim
        self.use_cross_view_vit = use_cross_view_vit
        self.use_semantic_cross_view_vit = use_semantic_cross_view_vit
        self.use_semantic_color_token_match = use_semantic_color_token_match
        self.use_polar_token_match = use_polar_token_match
        self.use_semantic_cnn_context = use_semantic_cnn_context
        self.use_street_object_aware = correction_type == "street_object_aware"
        self.use_satellite_color_bottleneck = correction_type == "satellite_color_bottleneck"
        self.use_satellite_detail_bottleneck = correction_type == "satellite_detail_bottleneck_v15"
        self.use_sky_gray_chroma_match = correction_type == "sky_gray_chroma_match_v20"
        self.semantic_context_encoder = None
        self.cross_view_feature_dim = cross_view_feature_dim
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Frozen DDColor
        self.ddcolor = DDColorWrapper(
            model_path=ddcolor_weights_path,
            model_code_path=ddcolor_code_path,
            input_size=256,
            device=self.device,
        )

        if self.use_sky_gray_chroma_match:
            self.satellite_encoder = None
            self.cross_view_encoder = None
            self.polar_encoder = None
            self.lane_vit = None
            self.correction = SkyOnlyGrayChromaMatchCorrectionModule(
                num_gray_regions=gray_region_bins,
                num_chroma_regions=chroma_region_bins,
                hidden_dim=street_object_hidden_dim,
                residual_scale=residual_scale,
                satellite_prior_strength=satellite_prior_strength,
                gray_temperature=gray_region_temperature,
                chroma_temperature=chroma_region_temperature,
                token_dim=token_selection_dim,
                detail_scale=street_object_detail_scale,
                street_semantic_source=street_semantic_source,
            )
            self.to(self.device)
            return

        if self.use_satellite_color_bottleneck or self.use_satellite_detail_bottleneck:
            self.satellite_encoder = None
            self.cross_view_encoder = None
            self.polar_encoder = None
            self.lane_vit = None
            correction_cls = (
                SatelliteDetailBottleneckCorrectionModule
                if self.use_satellite_detail_bottleneck
                else SatelliteColorBottleneckCorrectionModule
            )
            correction_kwargs = {}
            if self.use_satellite_detail_bottleneck:
                correction_kwargs = {
                    "lane_detail_strength": lane_detail_strength,
                    "satellite_dependency_boost": satellite_dependency_boost,
                    "lane_evidence_threshold": lane_evidence_threshold,
                }
            self.correction = correction_cls(
                num_semantic_classes=semantic_num_classes,
                hidden_dim=street_object_hidden_dim,
                residual_scale=residual_scale,
                detail_scale=street_object_detail_scale,
                satellite_prior_strength=satellite_prior_strength,
                use_street_gray_edges=use_street_gray_edges,
                use_street_gray_modulation=use_street_gray_modulation,
                use_gray_satellite_token_selection=use_gray_satellite_token_selection,
                use_satellite_chroma_token_selection=use_satellite_chroma_token_selection,
                token_selection_patch_size=token_selection_patch_size,
                token_selection_dim=token_selection_dim,
                street_semantic_source=street_semantic_source,
                satellite_semantic_source=satellite_semantic_source,
                **correction_kwargs,
            )
            self.to(self.device)
            return

        if self.use_street_object_aware:
            self.satellite_encoder = None
            self.cross_view_encoder = None
            self.polar_encoder = None
            self.lane_vit = None
            self.correction = StreetObjectAwareCorrectionModule(
                context_dim=context_dim,
                hidden_dim=street_object_hidden_dim,
                num_semantic_classes=semantic_num_classes,
                num_street_masks=street_object_num_masks,
                residual_scale=residual_scale,
                detail_scale=street_object_detail_scale,
                dino_model_name=dino_model_name,
                dino_pretrained=dino_pretrained,
            )
            self.to(self.device)
            return

        # 2. Satellite and Polar Context Encoders
        if use_semantic_cnn_context:
            self.satellite_encoder = None
            self.cross_view_encoder = None
            self.semantic_context_encoder = SemanticSatelliteStatsEncoder(
                context_dim=context_dim,
                num_classes=semantic_num_classes,
            )
        elif use_semantic_color_token_match or use_polar_token_match:
            self.satellite_encoder = None
            self.cross_view_encoder = SemanticColorTokenMatcher(
                street_channels=6,
                satellite_channels=3,
                num_classes=semantic_num_classes,
                embed_dim=cross_view_embed_dim,
                depth=cross_view_depth,
                num_heads=cross_view_heads,
                street_patch_size=cross_view_street_patch_size,
                satellite_patch_size=cross_view_satellite_patch_size,
                context_dim=context_dim,
                feature_channels=cross_view_feature_dim,
                image_size=256,
                context_size=polar_input_size if use_polar_token_match else None,
                color_weight=color_token_match_weight,
                semantic_distribution_weight=semantic_distribution_weight,
                boundary_weight=boundary_match_weight,
                token_delta_scale=token_delta_scale,
            )
        elif use_semantic_cross_view_vit:
            self.satellite_encoder = None
            self.cross_view_encoder = SemanticGuidedCrossViewViTEncoder(
                street_channels=6,
                satellite_channels=3,
                num_classes=semantic_num_classes,
                embed_dim=cross_view_embed_dim,
                depth=cross_view_depth,
                num_heads=cross_view_heads,
                street_patch_size=cross_view_street_patch_size,
                satellite_patch_size=cross_view_satellite_patch_size,
                context_dim=context_dim,
                feature_channels=cross_view_feature_dim,
                image_size=256,
            )
        elif use_cross_view_vit:
            self.satellite_encoder = None
            self.cross_view_encoder = CrossViewViTEncoder(
                street_channels=6,
                satellite_channels=3,
                embed_dim=cross_view_embed_dim,
                depth=cross_view_depth,
                num_heads=cross_view_heads,
                patch_size=cross_view_patch_size,
                context_dim=context_dim,
                feature_channels=cross_view_feature_dim,
                image_size=256,
            )
        elif use_satellite_vit:
            self.cross_view_encoder = None
            self.satellite_encoder = SatelliteViTColorEncoder(
                embed_dim=satellite_vit_embed_dim,
                depth=satellite_vit_depth,
                num_heads=satellite_vit_heads,
                patch_size=satellite_vit_patch_size,
                context_dim=context_dim,
                feature_channels=satellite_vit_feature_dim,
                image_size=256,
            )
        else:
            self.cross_view_encoder = None
            self.satellite_encoder = PolarContextEncoder(
                context_dim=context_dim,
                pretrained=satellite_encoder_pretrained,
                freeze_backbone=False,
            )
        if use_polar_context:
            self.polar_encoder = PolarContextEncoder(
                context_dim=context_dim,
                pretrained=polar_encoder_pretrained,
                freeze_backbone=False,  # We want to train this
            )
        else:
            self.polar_encoder = None

        # 3. Direction conditioning for the four panorama patches.
        self.direction_embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=context_dim,
            padding_idx=0,
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(context_dim * (3 if use_polar_context else 2), context_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(context_dim, context_dim),
        )

        # 4. Optional ViT detail branch for road/lane marking structures.
        if use_lane_vit:
            self.lane_vit = LaneViTDetailEncoder(
                input_channels=6,
                embed_dim=lane_vit_embed_dim,
                depth=lane_vit_depth,
                num_heads=lane_vit_heads,
                patch_size=lane_vit_patch_size,
                output_channels=lane_feature_dim,
                image_size=256,
            )
        else:
            self.lane_vit = None

        # 5. Residual Correction Module
        if use_semantic_cnn_context:
            self.correction = SemanticAwareCorrectionModule(
                context_dim=context_dim,
                hidden_dim=96,
                semantic_channels=semantic_num_classes,
                residual_scale=residual_scale,
            )
        elif use_semantic_color_token_match or use_polar_token_match:
            self.correction = TokenColorCorrectionModule(
                context_dim=context_dim,
                hidden_dim=64,
                detail_channels=cross_view_feature_dim,
                residual_scale=residual_scale,
                token_delta_scale=token_correction_scale,
            )
        elif use_lane_vit or use_satellite_vit or use_cross_view_vit or use_semantic_cross_view_vit:
            self.correction = DetailAwareLightCorrectionModule(
                context_dim=context_dim,
                hidden_dim=64,
                detail_channels=(
                    lane_feature_dim
                    if use_lane_vit
                    else cross_view_feature_dim
                    if (use_cross_view_vit or use_semantic_cross_view_vit)
                    else satellite_vit_feature_dim
                ),
                residual_scale=residual_scale,
            )
        elif correction_type == "light":
            self.correction = LightCorrectionModule(
                context_dim=context_dim,
                residual_scale=residual_scale,
            )
        else:
            self.correction = ResidualCorrectionModule(
                context_dim=context_dim,
                residual_scale=residual_scale,
                use_film=use_film,
            )

        # Move to device
        self.to(self.device)

    def forward(
        self,
        gray_rgb: torch.Tensor,
        polar_img: torch.Tensor,
        satellite_img: torch.Tensor = None,
        patch_idx: torch.Tensor = None,
        polar_seg: torch.Tensor = None,
        street_semantic: torch.Tensor = None,
        satellite_semantic: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            gray_rgb: Grayscale image in RGB format (B, 3, H, W), range [0, 1]
            polar_img: Polar satellite view (B, 3, H_p, W_p), range [0, 1]
            satellite_img: Original satellite view (B, 3, H_s, W_s), range [0, 1]
            patch_idx: Panorama patch index tensor (B,), values 1-4

        Returns:
            Dictionary with:
                - 'base_rgb': DDColor base output
                - 'context_vector': Context from polar encoder
                -final_rgb': Final corrected output
                'delta_color': Color correction delta
        """
        # 1. Get base colorization from DDColor (frozen)
        with torch.no_grad():
            base_rgb = self.ddcolor.colorize(gray_rgb).clone()

        if self.use_satellite_color_bottleneck or self.use_satellite_detail_bottleneck or self.use_sky_gray_chroma_match:
            result = self.correction(
                base_rgb,
                gray_rgb,
                satellite_img,
                street_semantic=street_semantic,
                satellite_semantic=satellite_semantic,
            )
            return {
                'base_rgb': base_rgb,
                'satellite_context': None,
                'polar_context': None,
                'direction_context': None,
                'context_vector': None,
                'satellite_detail_features': None,
                'satellite_color_prior': result.get('satellite_color_prior'),
                'satellite_class_color_prior': result.get('satellite_class_color_prior'),
                'satellite_token_attention': result.get('satellite_token_attention'),
                'token_delta': None,
                'cross_view_no_match_map': None,
                'cross_view_attention': None,
                'semantic_street_map': result.get('street_semantic'),
                'semantic_satellite_map': result.get('satellite_semantic'),
                'gray_region_probs': result.get('gray_region_probs'),
                'satellite_chroma_regions': result.get('satellite_chroma_regions'),
                'satellite_semantic_stats': result.get('satellite_color_prototypes'),
                'lane_detail_features': result.get('street_lane_candidate'),
                'lane_color_prior': result.get('satellite_lane_prior'),
                'object_attention': result.get('satellite_gate'),
                'street_masks': result.get('street_semantic'),
                'global_delta': result.get('satellite_delta'),
                'detail_delta': (
                    result.get('line_detail_delta')
                    if result.get('lane_delta') is None
                    else result.get('line_detail_delta') + result.get('lane_delta')
                ),
                'lane_delta': result.get('lane_delta'),
                'lane_gate': result.get('lane_gate'),
                'satellite_lane_evidence': result.get('satellite_lane_evidence'),
                'satellite_lane_mask': result.get('satellite_lane_mask'),
                'satellite_lane_prior': result.get('satellite_lane_prior'),
                'preservation_gate': result.get('satellite_gate'),
                'street_edge': result.get('street_edge'),
                'satellite_edge': result.get('satellite_edge'),
                'semantic_boundary': result.get('semantic_boundary'),
                'sky_mask': result.get('sky_mask'),
                'car_mask': result.get('car_mask'),
                'street_only_mask': result.get('street_only_mask'),
                'non_sky_mask': result.get('non_sky_mask'),
                'satellite_guided_mask': result.get('satellite_guided_mask'),
                'satellite_gate': result.get('satellite_gate'),
                'final_rgb': result['final_rgb'],
                'delta_color': result['delta_color'],
            }

        if self.use_street_object_aware:
            result = self.correction(
                base_rgb,
                gray_rgb,
                polar_img,
                polar_seg_rgb=polar_seg,
                satellite_semantic=satellite_semantic,
                street_semantic=street_semantic,
                patch_idx=patch_idx,
            )
            return {
                'base_rgb': base_rgb,
                'satellite_context': result.get('prior_context'),
                'polar_context': result.get('prior_context'),
                'direction_context': None,
                'context_vector': result.get('prior_context'),
                'satellite_detail_features': None,
                'satellite_color_prior': None,
                'token_delta': None,
                'cross_view_no_match_map': None,
                'cross_view_attention': None,
                'semantic_street_map': result.get('street_masks'),
                'semantic_satellite_map': satellite_semantic,
                'satellite_semantic_stats': None,
                'lane_detail_features': None,
                'lane_color_prior': None,
                'object_attention': result.get('object_attention'),
                'street_masks': result.get('street_masks'),
                'global_delta': result.get('global_delta'),
                'detail_delta': result.get('detail_delta'),
                'preservation_gate': result.get('preservation_gate'),
                'final_rgb': result['final_rgb'],
                'delta_color': result['delta_color'],
            }

        # 2. Extract satellite, polar, and direction context.
        if satellite_img is None:
            satellite_img = torch.zeros(
                gray_rgb.size(0),
                3,
                256,
                256,
                device=gray_rgb.device,
                dtype=gray_rgb.dtype,
            )
        satellite_detail_features = None
        satellite_color_prior = None
        token_delta = None
        cross_view_no_match_map = None
        cross_view_attention = None
        semantic_street_map = None
        semantic_satellite_map = None
        satellite_semantic_stats = None
        if self.semantic_context_encoder is not None:
            semantic_result = self.semantic_context_encoder(satellite_img, satellite_semantic)
            satellite_context = semantic_result["context"]
            semantic_satellite_map = semantic_result["satellite_semantic"]
            satellite_semantic_stats = semantic_result.get("stats")
            semantic_street_map = self.semantic_context_encoder.street_one_hot(
                gray_rgb, base_rgb, street_semantic
            )
        elif self.cross_view_encoder is not None:
            if self.use_semantic_cross_view_vit or self.use_semantic_color_token_match or self.use_polar_token_match:
                context_img = polar_img if self.use_polar_token_match else satellite_img
                context_semantic = None if self.use_polar_token_match else satellite_semantic
                satellite_result = self.cross_view_encoder(
                    gray_rgb,
                    base_rgb,
                    context_img,
                    street_semantic=street_semantic,
                    satellite_semantic=context_semantic,
                )
            else:
                satellite_result = self.cross_view_encoder(gray_rgb, base_rgb, satellite_img)
            satellite_context = satellite_result["context"]
            satellite_detail_features = satellite_result.get("features")
            satellite_color_prior = satellite_result.get("color_prior")
            token_delta = satellite_result.get("token_delta")
            cross_view_no_match_map = satellite_result.get("no_match_map")
            cross_view_attention = satellite_result.get("cross_attention")
        else:
            satellite_result = self.satellite_encoder(satellite_img)
            if isinstance(satellite_result, dict):
                satellite_context = satellite_result["context"]
                satellite_detail_features = satellite_result.get("features")
                satellite_color_prior = satellite_result.get("color_prior")
            else:
                satellite_context = satellite_result
        if self.polar_encoder is not None:
            polar_context = self.polar_encoder(polar_img)
        else:
            polar_context = None
        if patch_idx is None:
            patch_idx = torch.zeros(gray_rgb.size(0), device=gray_rgb.device, dtype=torch.long)
        else:
            patch_idx = patch_idx.to(device=gray_rgb.device, dtype=torch.long).clamp(0, 4)
        direction_context = self.direction_embedding(patch_idx)
        context_parts = [satellite_context]
        if polar_context is not None:
            context_parts.append(polar_context)
        context_parts.append(direction_context)
        context_vector = self.context_fusion(torch.cat(context_parts, dim=1))

        detail_features = satellite_detail_features
        color_prior = satellite_color_prior
        lane_detail_features = None
        lane_color_prior = None
        if self.lane_vit is not None:
            lane_result = self.lane_vit(gray_rgb, base_rgb)
            lane_detail_features = lane_result["lane_detail_features"]
            lane_color_prior = lane_result["lane_color_prior"]
            detail_features = lane_detail_features
            color_prior = lane_color_prior

        # 3. Apply residual correction
        if isinstance(self.correction, SemanticAwareCorrectionModule):
            result = self.correction(
                base_rgb,
                context_vector,
                gray_input=gray_rgb,
                street_semantic=semantic_street_map,
            )
        elif isinstance(self.correction, TokenColorCorrectionModule):
            result = self.correction(
                base_rgb,
                context_vector,
                detail_features=detail_features,
                color_prior=color_prior,
                token_delta=token_delta,
            )
        elif isinstance(self.correction, DetailAwareLightCorrectionModule):
            result = self.correction(
                base_rgb,
                context_vector,
                detail_features=detail_features,
                color_prior=color_prior,
            )
        else:
            result = self.correction(base_rgb, context_vector)

        return {
            'base_rgb': base_rgb,
            'satellite_context': satellite_context,
            'polar_context': polar_context,
            'direction_context': direction_context,
            'context_vector': context_vector,
            'satellite_detail_features': satellite_detail_features,
            'satellite_color_prior': satellite_color_prior,
            'token_delta': token_delta,
            'cross_view_no_match_map': cross_view_no_match_map,
            'cross_view_attention': cross_view_attention,
            'semantic_street_map': semantic_street_map,
            'semantic_satellite_map': semantic_satellite_map,
            'satellite_semantic_stats': satellite_semantic_stats,
            'lane_detail_features': lane_detail_features,
            'lane_color_prior': lane_color_prior,
            'final_rgb': result['final_rgb'],
            'delta_color': result['delta_color'],
        }

    def get_trainable_parameters(self):
        """Get list of trainable parameters (excluding DDColor)."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_num_parameters(self) -> dict:
        """Return parameter statistics."""
        ddcolor_total, ddcolor_frozen = self.ddcolor.get_num_parameters()

        trainable_params = []
        frozen_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param.numel()))
            else:
                frozen_params.append((name, param.numel()))

        total_trainable = sum(p[1] for p in trainable_params)
        total_frozen = sum(p[1] for p in frozen_params)

        return {
            'ddcolor_total': ddcolor_total,
            'ddcolor_frozen': ddcolor_frozen,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'total_trainable': total_trainable,
            'total_frozen': total_frozen,
            'total': total_trainable + total_frozen,
        }

    def verify_ddcolor_frozen(self) -> bool:
        """Verify DDColor weights are frozen."""
        return self.ddcolor.verify_frozen()
