# models/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Liquid_Fusion_Cell(nn.Module):
    """
    [Core Module] Liquid Fusion Cell (Based on CfC Theory)

    Theoretical Basis:
    Instead of static feature addition, this module approximates the
    Closed-form Continuous-depth (CfC) solution of a Liquid Time-Constant (LTC) system.

    Dynamics:
    - h (Global Stream): Modeled as the 'Liquid State' x(t).
    - l (Local Stream): Modeled as the 'Synaptic Stimuli' I(t).
    - Fusion: Implements a 'Time-Continuous Gating' mechanism that balances
      state retention (memory) and stimulus injection (sensory update).

    Equation: x(t) = (1 - Gate) * x(t-1) + Gate * I(t)
    """

    def __init__(self, in_dim_h, in_dim_l, out_dim):
        super(Liquid_Fusion_Cell, self).__init__()

        # 1. State Alignment (Dimension Matching)
        self.state_trans = nn.Sequential(
            BasicConv2d(in_dim_h, out_dim, 1),
            BasicConv2d(out_dim, out_dim, 3, padding=1)
        )
        self.input_trans = BasicConv2d(in_dim_l, out_dim, 1)

        # 2. Solute Concentration Control (Channel Attention)
        # Determines the reactivity of the liquid state to specific synaptic features.
        # This replaces the generic channel attention with a state-dependent filter.
        self.concentration_mlp = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // 4, out_dim, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # 3. Liquid Time-Constant Gate (Spatial Gating)
        # Inspired by CfC Equation (4): x(t) = sigma * g(x) + (1-sigma) * h(x).
        # This gate acts as the "Time Constant" (tau), controlling the 'permeability'
        # or how fast the local details (l) diffuse into the global state (h).
        self.liquid_gate = nn.Sequential(
            BasicConv2d(out_dim * 2, out_dim, 3, padding=1),
            nn.Conv2d(out_dim, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.smooth = BasicConv2d(out_dim, out_dim, 3, padding=1)
        self._init_weights()

    def _init_weights(self):
        # Initialize input stream normalization to 0 to ensure stability at epoch 0
        nn.init.constant_(self.input_trans.bn.weight, 0)
        nn.init.constant_(self.input_trans.bn.bias, 0)

    def forward(self, h, l):
        """
        h: Global Stream (The Liquid State / Context)
        l: Local Stream (The Synaptic Input / Stimuli)
        """
        # --- 1. Alignment ---
        if h.size()[2:] != l.size()[2:]:
            l = F.interpolate(l, size=h.size()[2:], mode='bilinear', align_corners=False)

        state_feat = self.state_trans(h)  # x(t)
        input_feat = self.input_trans(l)  # I(t)

        # --- 2. Concentration Control (Channel Osmosis) ---
        # The state (h) determines which channels of the input (l) are relevant.
        # This mimics the nonlinear synaptic transmission S(t).
        pool_state = state_feat
        avg_out = self.concentration_mlp(F.avg_pool2d(pool_state, pool_state.size(2)))
        max_out = self.concentration_mlp(F.adaptive_max_pool2d(pool_state, 1))

        reactivity = self.sigmoid(avg_out + max_out)
        input_refined = input_feat * reactivity

        # --- 3. Liquid Gating (Closed-form Dynamics) ---
        # We model the fusion as a dynamic equilibrium.
        # gate (sigma) represents the "permeability" or inverse time-constant.

        # Calculate the gating factor based on both State and Input
        gate_map = self.liquid_gate(torch.cat([state_feat, input_refined], dim=1))

        # [Core Logic Upgrade] CfC-inspired Dynamic Equilibrium:
        # Instead of simple addition, we use "Gated Injection" with Decay.
        # Logic: If permeability (gate) is high, input enters and state updates.
        #        If permeability is low, state is preserved (memory).
        # Equation matches CfC Eq.4: x(t) = (1-sigma)*h(x) + sigma*g(x)
        injected_state = state_feat * (1.0 - gate_map) + input_refined * gate_map

        return self.smooth(injected_state)


class SaliencyGuidedUpsample(nn.Module):
    """
    [Final Optimized] Saliency Guided Upsample (Spectral-Spatial Co-Design)

    Design for SOD (Salient Object Detection):
    1. Spatial Branch: Enhances high-frequency boundary details (Edges).
    2. Spectral Branch: Captures global shape and semantic integrity via FFT (Body).
    3. AMP Safety: Explicitly handles float32 casting for FFT stability.
    """

    def __init__(self, in_channels, scale=2):
        super(SaliencyGuidedUpsample, self).__init__()
        self.scale = scale
        self.in_channels = in_channels

        # 1. Spatial Branch (Local Details)
        # Uses standard convolution to preserve edges and local texture
        self.spatial_process = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 3, padding=1),
            BasicConv2d(in_channels, in_channels, 3, padding=1)
        )

        # 2. Spectral Branch (Global Context)
        # Parameters for the learnable spectral filter
        self.base_freq_h = 32
        self.base_freq_w = 16  # Corresponds to width/2 + 1 in RFFT

        # Learnable Complex Weights: [1, C, H_base, W_base, 2]
        # We initialize this as a parameter map to be interpolated later
        self.spectral_weight = nn.Parameter(
            torch.empty(1, in_channels, self.base_freq_h, self.base_freq_w, 2, dtype=torch.float32)
        )

        # 3. Fusion & Reconstruction
        # Fuses the dual-domain features and reduces channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            BasicConv2d(in_channels, in_channels, 3, padding=1)
        )

        # Residual connection to ensure gradient flow at epoch 0
        self.skip_conv = BasicConv2d(in_channels, in_channels, 1)

        self._init_spectral_weights()

    def _init_spectral_weights(self):
        # Truncated normal initialization for stability
        # Keeps values small so the spectral branch gradually learns to "activate"
        nn.init.trunc_normal_(self.spectral_weight, std=0.02)

    def forward(self, x):
        # 1. Base Spatial Upsampling (Bilinear)
        x_up = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        # --- Branch A: Spatial Processing (Boundary Sharpening) ---
        x_spatial = self.spatial_process(x_up)

        # --- Branch B: Spectral Processing (Global Integrity) ---
        # [CRITICAL] AMP Safety: Force float32 for FFT operations
        # This prevents "ComplexHalf" warnings and numerical instability (NaNs)
        x_up_32 = x_up.float()

        # B.1 FFT (Real -> Complex)
        # x_fft shape: [B, C, H, W/2 + 1]
        x_fft = torch.fft.rfft2(x_up_32, norm='backward')
        B, C, H, W_freq = x_fft.shape

        # B.2 Dynamic Spectral Filter Generation
        # Interpolate the base learnable weights to match current resolution
        # weight_permuted shape logic:
        # [1, C, H_base, W_base, 2] -> permute -> [1, 2, C, H_base, W_base] -> reshape -> [1, 2*C, H_base, W_base]
        # We cast weight to float32 explicitly to match x_up_32
        weight_in = self.spectral_weight.to(dtype=torch.float32)
        weight_permuted = weight_in.permute(0, 4, 1, 2, 3).reshape(1, 2 * C, self.base_freq_h, self.base_freq_w)

        # Interpolate to current [H, W_freq]
        current_filter = F.interpolate(
            weight_permuted,
            size=(H, W_freq),
            mode='bilinear',
            align_corners=False
        )

        # Reshape back to complex format: [1, C, H, W_freq, 2]
        current_filter = current_filter.view(1, 2, C, H, W_freq).permute(0, 2, 3, 4, 1)
        filter_complex = torch.view_as_complex(current_filter.contiguous())

        # B.3 Spectral Gating (Hadamard Product)
        # Modulates frequencies to highlight salient objects
        x_gated_fft = x_fft * filter_complex

        # B.4 IFFT (Complex -> Real)
        x_spectral_32 = torch.fft.irfft2(x_gated_fft, s=(x_up.size(2), x_up.size(3)), norm='backward')

        # [CRITICAL] Cast back to original precision (e.g., float16) for fusion
        x_spectral = x_spectral_32.type_as(x_spatial)

        # 4. Dual-Domain Fusion
        combined = torch.cat([x_spatial, x_spectral], dim=1)
        out = self.fuse(combined)

        # Residual Addition
        return out + self.skip_conv(x_up)


class FinalRefinement(nn.Module):
    def __init__(self, in_channels):
        super(FinalRefinement, self).__init__()
        self.refine = nn.Sequential(
            BasicConv2d(in_channels, in_channels, 3, padding=1),
            BasicConv2d(in_channels, in_channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.refine(x)