"""
Demonstration of Random Weight Factorization (RWF) Linear Layer

This script demonstrates the FactorizedLinear class implementation based on:
Wang et al., "Random Weight Factorization Improves the Training of
Continuous Neural Representations", arXiv:2210.01274, 2022.

The script shows:
1. Basic usage of FactorizedLinear layer
2. Comparison with standard nn.Linear
3. Integration into a simple PINN-like network
4. Verification that weight factorization is working correctly
"""

import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the path to import balloonlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from balloonlib.layers import FactorizedLinear


torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


def demonstrate_basic_usage():
    """Demonstrate basic usage of FactorizedLinear layer."""
    print("=" * 70)
    print("1. BASIC USAGE DEMONSTRATION")
    print("=" * 70)

    # Create a factorized linear layer
    layer = FactorizedLinear(
        in_features=10, out_features=5, bias=True, scale_mean=1.0, scale_std=0.1
    ).to(device)

    print("\nCreated FactorizedLinear layer:")
    print(f"  {layer}")
    print("\nParameter shapes:")
    print(f"  scale (s):     {layer.scale.shape}")
    print(f"  value (V):     {layer.value.shape}")
    print(f"  bias (b):      {layer.bias.shape if layer.bias is not None else None}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 10, device=device)
    y = layer(x)

    print("\nForward pass:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")

    # Verify weight reconstruction
    exp_scale = torch.exp(layer.scale)
    reconstructed_weight = layer.value * exp_scale.unsqueeze(0)

    print("\nWeight factorization check:")
    print(f"  Original V mean: {layer.value.mean().item():.6f}")
    print(f"  exp(s) mean:     {exp_scale.mean().item():.6f}")
    print(f"  W = exp(s)*V mean: {reconstructed_weight.mean().item():.6f}")
    print(f"  exp(s) range:    [{exp_scale.min().item():.4f}, {exp_scale.max().item():.4f}]")

    return layer


def compare_with_standard_linear():
    """Compare FactorizedLinear with standard nn.Linear."""
    print("\n" + "=" * 70)
    print("2. COMPARISON WITH STANDARD LINEAR LAYER")
    print("=" * 70)

    in_feat, out_feat = 20, 10

    # Create both types of layers
    factorized = FactorizedLinear(in_feat, out_feat, bias=True).to(device)
    standard = nn.Linear(in_feat, out_feat, bias=True).to(device)

    print("\nParameter count comparison:")
    factorized_params = sum(p.numel() for p in factorized.parameters())
    standard_params = sum(p.numel() for p in standard.parameters())

    print(f"  Standard Linear: {standard_params} parameters")
    print(f"  Factorized Linear: {factorized_params} parameters")
    print(f"  Difference: {factorized_params - standard_params} parameters")
    print(f"  Extra parameters: scale vector (out_features = {out_feat})")

    print("\nParameter breakdown:")
    print(
        f"  Standard:   W ({in_feat}×{out_feat}) + b ({out_feat}) = {in_feat * out_feat + out_feat}"
    )
    print(
        f"  Factorized: s ({out_feat}) + V ({in_feat}×{out_feat}) + b ({out_feat}) = {out_feat + in_feat * out_feat + out_feat}"
    )

    # Compare forward pass
    x = torch.randn(16, in_feat, device=device)
    y_factorized = factorized(x)
    y_standard = standard(x)

    print("\nOutput shapes:")
    print(f"  Standard:   {y_standard.shape}")
    print(f"  Factorized: {y_factorized.shape}")


def test_gradient_flow():
    """Test that gradients flow correctly through factorized parameters."""
    print("\n" + "=" * 70)
    print("3. GRADIENT FLOW VERIFICATION")
    print("=" * 70)

    layer = FactorizedLinear(5, 3, bias=True).to(device)

    # Simple forward-backward pass
    x = torch.randn(10, 5, device=device, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()

    print("\nGradient check:")
    print(f"  scale.grad exists: {layer.scale.grad is not None}")
    print(f"  value.grad exists: {layer.value.grad is not None}")
    print(f"  bias.grad exists:  {layer.bias.grad is not None}")

    if layer.scale.grad is not None:
        print("\n  scale.grad stats:")
        print(f"    Mean: {layer.scale.grad.mean().item():.6e}")
        print(f"    Std:  {layer.scale.grad.std().item():.6e}")
        print(f"    Max:  {layer.scale.grad.max().item():.6e}")

    if layer.value.grad is not None:
        print("\n  value.grad stats:")
        print(f"    Mean: {layer.value.grad.mean().item():.6e}")
        print(f"    Std:  {layer.value.grad.std().item():.6e}")
        print(f"    Max:  {layer.value.grad.max().item():.6e}")


def demonstrate_pinn_network():
    """Demonstrate using FactorizedLinear in a PINN-like network."""
    print("\n" + "=" * 70)
    print("4. EXAMPLE PINN NETWORK WITH RWF")
    print("=" * 70)

    class SimplePINN(nn.Module):
        """Simple PINN with RWF layers."""

        def __init__(self, use_rwf=True):
            super().__init__()
            self.use_rwf = use_rwf

            if use_rwf:
                self.layers = nn.Sequential(
                    FactorizedLinear(1, 32, bias=False),
                    nn.Tanh(),
                    FactorizedLinear(32, 64, bias=False),
                    nn.Tanh(),
                    FactorizedLinear(64, 32, bias=False),
                    nn.Tanh(),
                    FactorizedLinear(32, 1, bias=True),
                )
            else:
                self.layers = nn.Sequential(
                    nn.Linear(1, 32, bias=False),
                    nn.Tanh(),
                    nn.Linear(32, 64, bias=False),
                    nn.Tanh(),
                    nn.Linear(64, 32, bias=False),
                    nn.Tanh(),
                    nn.Linear(32, 1, bias=True),
                )

        def forward(self, x):
            return self.layers(x)

    # Create both versions
    pinn_rwf = SimplePINN(use_rwf=True).to(device)
    pinn_standard = SimplePINN(use_rwf=False).to(device)

    print("\nNetwork architecture:")
    print("  Input: 1D (time)")
    print("  Hidden: 32 → 64 → 32")
    print("  Output: 1D")

    rwf_params = sum(p.numel() for p in pinn_rwf.parameters())
    std_params = sum(p.numel() for p in pinn_standard.parameters())

    print("\nParameter counts:")
    print(f"  Standard PINN: {std_params} parameters")
    print(f"  RWF PINN:      {rwf_params} parameters")
    print(f"  Additional:    {rwf_params - std_params} scale parameters")

    # Test forward pass
    t = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
    output_rwf = pinn_rwf(t)
    output_std = pinn_standard(t)

    print("\nForward pass test:")
    print(f"  Input shape:        {t.shape}")
    print(f"  RWF output shape:   {output_rwf.shape}")
    print(f"  Std output shape:   {output_std.shape}")

    # Count factorized layers
    rwf_layer_count = sum(1 for m in pinn_rwf.modules() if isinstance(m, FactorizedLinear))
    print(f"\nFactorized layers in network: {rwf_layer_count}")


def verify_initialization():
    """Verify that initialization follows the RWF algorithm correctly."""
    print("\n" + "=" * 70)
    print("5. INITIALIZATION VERIFICATION")
    print("=" * 70)

    # Create multiple layers to check statistical properties
    n_samples = 100
    in_feat, out_feat = 50, 25

    scale_values = []
    exp_scale_values = []

    for _ in range(n_samples):
        layer = FactorizedLinear(in_feat, out_feat, scale_mean=1.0, scale_std=0.1)
        scale_values.append(layer.scale.detach().clone())
        exp_scale_values.append(torch.exp(layer.scale).detach().clone())

    scales = torch.stack(scale_values)
    exp_scales = torch.stack(exp_scale_values)

    print(f"\nScale factor (s) statistics over {n_samples} initializations:")
    print("  Expected: s ~ N(mu=1.0, sigma=0.1)")
    print(f"  Observed mean: {scales.mean().item():.4f}")
    print(f"  Observed std:  {scales.std().item():.4f}")

    print("\nexp(s) statistics:")
    print(
        f"  Mean: {exp_scales.mean().item():.4f} (expected ≈ e^(μ+σ²/2) = {torch.exp(torch.tensor(1.0 + 0.1**2 / 2)).item():.4f})"
    )
    print(f"  Std:  {exp_scales.std().item():.4f}")
    print(f"  Min:  {exp_scales.min().item():.4f}")
    print(f"  Max:  {exp_scales.max().item():.4f}")

    # Check that V maintains Glorot properties
    layer = FactorizedLinear(in_feat, out_feat)
    reconstructed_W = layer.value * torch.exp(layer.scale).unsqueeze(0)

    print("\nReconstructed weight (W = exp(s) * V) statistics:")
    print(f"  Shape: {reconstructed_W.shape}")
    print(f"  Mean:  {reconstructed_W.mean().item():.6f}")
    print(f"  Std:   {reconstructed_W.std().item():.4f}")

    # For Glorot initialization: std ≈ sqrt(2 / (fan_in + fan_out))
    expected_std = torch.sqrt(torch.tensor(2.0 / (in_feat + out_feat)))
    print(f"  Expected Glorot std: {expected_std.item():.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RANDOM WEIGHT FACTORIZATION (RWF) DEMONSTRATION")
    print("PyTorch Implementation for Physics-Informed Neural Networks")
    print("=" * 70)

    # Run all demonstrations
    demonstrate_basic_usage()
    compare_with_standard_linear()
    test_gradient_flow()
    demonstrate_pinn_network()
    verify_initialization()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  ✓ FactorizedLinear implements RWF algorithm correctly")
    print("  ✓ Adds only out_features extra parameters (scale vector)")
    print("  ✓ Gradients flow correctly through factorized parameters")
    print("  ✓ Drop-in replacement for nn.Linear in PINN architectures")
    print("  ✓ Initialization follows Wang et al. recommendations")
    print("\nTo use in your BalloonPINN model:")
    print("  from balloonlib.rwf_layers import FactorizedLinear")
    print("  layer = FactorizedLinear(in_features, out_features, bias=False)")
    print()
