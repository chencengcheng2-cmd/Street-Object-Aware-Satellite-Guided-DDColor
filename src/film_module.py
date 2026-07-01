"""
FiLM (Feature-wise Linear Modulation) module for conditioning features with context.
"""

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """Generate FiLM parameters (gamma, beta) from context vector."""

    def __init__(
        self,
        context_dim: int,
        feature_dims: list,
        hidden_dim: int = 256,
        use_sequential: bool = False,
    ):
        """
        Args:
            context_dim: Dimension of context vector
            feature_dims: List of feature channel dimensions to generate FiLM for
            hidden_dim: Hidden layer dimension
            use_sequential: If True, generate separate FiLM for each layer sequentially
        """
        super().__init__()
        self.context_dim = context_dim
        self.feature_dims = feature_dims
        self.use_sequential = use_sequential

        if use_sequential:
            # Separate generator for each feature dimension
            self.generators = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(context_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, dim * 2),  # gamma and beta
                )
                for dim in feature_dims
            ])
        else:
            # Shared generator
            self.net = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, max(feature_dims) * 2),
            )

    def forward(self, context: torch.Tensor) -> list:
        """
        Generate FiLM parameters.

        Args:
            context: Context vector (B, context_dim)

        Returns:
            List of (gamma, beta) tuples, one for each feature dimension
        """
        if self.use_sequential:
            return [self._generate_films(gen(context), dim)
                    for gen, dim in zip(self.generators, self.feature_dims)]
        else:
            films = self.net(context)  # (B, max_dim * 2)
            result = []
            offset = 0
            for dim in self.feature_dims:
                gamma_beta = films[:, offset:offset + dim * 2]
                gamma, beta = self._split_films(gamma_beta)
                result.append((gamma, beta))
                offset += dim * 2
            return result

    def _generate_films(self, films: torch.Tensor, dim: int) -> tuple:
        """Split films into gamma and beta."""
        gamma, beta = films.chunk(2, dim=1)
        return gamma, beta

    def _split_films(self, films: torch.Tensor) -> tuple:
        """Split films into gamma and beta."""
        return films.chunk(2, dim=1)


class FiLMModulation(nn.Module):
    """Apply FiLM modulation to features."""

    def __init__(
        self,
        feature_dim: int,
        init_gamma: float = 0.0,
        init_beta: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Learnable bias for initialization
        self.register_buffer('init_gamma', torch.tensor(init_gamma))
        self.register_buffer('init_beta', torch.tensor(init_beta))

    def forward(
        self,
        features: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            features: Input features (B, C, H, W)
            gamma: Scale parameter (B, C, 1, 1) or broadcastable
            beta: Shift parameter (B, C, 1, 1) or broadcastable

        Returns:
            Modulated features (B, C, H, W)
        """
        # Ensure gamma and beta have correct shape
        if gamma.dim() == 2:  # (B, C)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        if beta.dim() == 2:
            beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Apply modulation: F' = (1 + gamma) * F + beta
        modulated = (1 + gamma) * features + beta
        return modulated


class FiLMLayer(nn.Module):
    """Combined FiLM generator and modulation layer."""

    def __init__(
        self,
        context_dim: int,
        feature_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        self.modulation = FiLMModulation(feature_dim)
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """Generate and apply FiLM modulation."""
        films = self.generator(context)
        gamma, beta = films.chunk(2, dim=1)
        return self.modulation(features, gamma, beta)


if __name__ == "__main__":
    # Test FiLM
    context = torch.randn(4, 512)
    features = torch.randn(4, 256, 32, 32)

    # Test FiLMLayer
    film_layer = FiLMLayer(512, 256)
    modulated = film_layer(features, context)
    print(f"Input features: {features.shape}")
    print(f"Modulated features: {modulated.shape}")

    # Test FiLMGenerator with multiple feature dims
    generator = FiLMGenerator(512, [64, 128, 256])
    films = generator(context)
    print(f"Generated {len(films)} FiLM parameter sets")
    for i, (g, b) in enumerate(films):
        print(f"  Set {i}: gamma={g.shape}, beta={b.shape}")
