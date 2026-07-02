
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class ConvBNReLU(nn.Module):
    """
    Convolution -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class DepthwiseConv(nn.Module):
    """
    Depthwise Separable Convolution
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            # Depthwise
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False
            ),

            nn.BatchNorm2d(in_channels),

            nn.ReLU(inplace=True),

            # Pointwise
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
class NuclearDensityEstimator(nn.Module):
    """
    Estimates a soft nuclear density map from encoder features.
    The output is a single-channel attention map highlighting
    regions that likely contain dense nuclei.
    """

    def __init__(self, in_channels=512, reduction=4):
        super().__init__()

        reduced = in_channels // reduction

        # Channel reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, reduced, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True)
        )

        # Fine texture branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(reduced, reduced,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True)
        )

        # Larger tissue context
        self.branch5 = nn.Sequential(
            nn.Conv2d(reduced,
                      reduced,
                      kernel_size=5,
                      padding=2,
                      bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced * 2,
                      reduced,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),

            nn.Conv2d(reduced,
                      1,
                      kernel_size=1)
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):

        x = self.reduce(x)

        fine = self.branch3(x)

        coarse = self.branch5(x)

        fusion = torch.cat([fine, coarse], dim=1)

        density = self.fusion(fusion)

        density = self.activation(density)

        return density

class TissuePatternExtractor(nn.Module):
    """
    Multi-branch tissue feature extraction.
    Each branch focuses on a different histopathological pattern.
    """

    def __init__(self, in_channels=512):

        super().__init__()

        branch_channels = in_channels // 4

        # Fine texture
        self.texture = nn.Sequential(
            nn.Conv2d(in_channels,
                      branch_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Cell clusters
        self.cluster = nn.Sequential(
            nn.Conv2d(in_channels,
                      branch_channels,
                      kernel_size=5,
                      padding=2,
                      bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Tumor boundaries
        self.boundary = nn.Sequential(
            nn.Conv2d(in_channels,
                      branch_channels,
                      kernel_size=3,
                      padding=2,
                      dilation=2,
                      bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Global tissue context
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      branch_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 4,
                      in_channels,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        B, C, H, W = x.shape

        texture = self.texture(x)

        cluster = self.cluster(x)

        boundary = self.boundary(x)

        global_feature = self.global_pool(x)
        global_feature = self.global_conv(global_feature)
        global_feature = F.interpolate(
            global_feature,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        features = torch.cat([
            texture,
            cluster,
            boundary,
            global_feature
        ], dim=1)

        out = self.fusion(features)

        return out

class AdaptiveCompetitiveFusion(nn.Module):
    """
    Adaptive branch competition.
    Learns how much each tissue branch should contribute.
    """

    def __init__(self,
                 channels=512,
                 num_branches=4,
                 reduction=16):

        super().__init__()

        self.num_branches = num_branches
        self.channels = channels

        hidden = max(channels // reduction, 32)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(channels * num_branches, hidden),

            nn.ReLU(inplace=True),

            nn.Linear(hidden, num_branches)

        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self,
                texture,
                cluster,
                boundary,
                global_context):

        B = texture.size(0)

        branches = [
            texture,
            cluster,
            boundary,
            global_context
        ]

        descriptors = []

        for feat in branches:

            d = self.gap(feat)

            d = d.view(B, -1)

            descriptors.append(d)

        descriptor = torch.cat(descriptors, dim=1)

        weights = self.fc(descriptor)

        weights = self.softmax(weights)

        fused = 0

        for i, feat in enumerate(branches):

            w = weights[:, i].view(B, 1, 1, 1)

            fused = fused + feat * w

        return fused, weights



class ConfidenceGuidedResidualFusion(nn.Module):
    """
    Confidence-guided fusion between original encoder features
    and ATPF-enhanced features.
    """

    def __init__(self,
                 channels=512):

        super().__init__()

        self.confidence = nn.Sequential(

            nn.Conv2d(
                channels + 1,
                channels // 4,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(channels // 4),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // 4,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()

        )

    def forward(self,
                original,
                enhanced,
                density):

        confidence = self.confidence(
            torch.cat([enhanced, density], dim=1)
        )

        output = confidence * enhanced + \
                 (1 - confidence) * original

        return output, confidence
    
class ATPFv2(nn.Module):
    """
    Adaptive Tissue Pattern Fusion Version 2
    """

    def __init__(self,
                 channels=512):

        super().__init__()

        ##################################################
        # Nuclear Density Estimation
        ##################################################

        self.density_estimator = NuclearDensityEstimator(
            in_channels=channels
        )

        ##################################################
        # Pattern Extractor
        ##################################################

        self.pattern_extractor = TissuePatternExtractor(
            in_channels=channels
        )

        ##################################################
        # Feature Branches
        ##################################################

        branch_channels = channels

        self.texture_branch = nn.Sequential(

            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(branch_channels),

            nn.ReLU(inplace=True)

        )

        self.cluster_branch = nn.Sequential(

            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=5,
                padding=2,
                bias=False
            ),

            nn.BatchNorm2d(branch_channels),

            nn.ReLU(inplace=True)

        )

        self.boundary_branch = nn.Sequential(

            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False
            ),

            nn.BatchNorm2d(branch_channels),

            nn.ReLU(inplace=True)

        )

        self.global_branch = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=1
            )

        )

        ##################################################
        # Competitive Fusion
        ##################################################

        self.acf = AdaptiveCompetitiveFusion(
            channels=channels,
            num_branches=4
        )

        ##################################################
        # Residual Fusion
        ##################################################

        self.cgrf = ConfidenceGuidedResidualFusion(
            channels=channels
        )

    def forward(self, x):

        ##################################################
        # Save Original Feature
        ##################################################

        original = x

        ##################################################
        # Density Map
        ##################################################

        density = self.density_estimator(x)

        ##################################################
        # Pattern Extraction
        ##################################################

        feature = self.pattern_extractor(x)

        ##################################################
        # Four Experts
        ##################################################

        texture = self.texture_branch(feature)

        cluster = self.cluster_branch(feature)

        boundary = self.boundary_branch(feature)

        global_context = self.global_branch(feature)

        global_context = F.interpolate(
            global_context,
            size=feature.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        ##################################################
        # Adaptive Competition
        ##################################################

        enhanced, branch_weights = self.acf(

            texture,

            cluster,

            boundary,

            global_context

        )

        ##################################################
        # Confidence Fusion
        ##################################################

        output, confidence = self.cgrf(

            original,

            enhanced,

            density

        )

        ##################################################
        # Useful outputs for visualization
        ##################################################

        extras = {

            "density_map": density,

            "confidence_map": confidence,

            "branch_weights": branch_weights

        }

        return output, extras

class UNetATPF(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = smp.Unet(

            encoder_name="resnet34",

            encoder_weights="imagenet",

            in_channels=3,

            classes=1

        )

        self.atpf = ATPFv2(
            channels=512
        )

    def forward(self,x):

        features = list(self.model.encoder(x))

        bottleneck = features[-1]

        enhanced, extras = self.atpf(bottleneck)

        features[-1] = enhanced

        decoder_output = self.model.decoder(features)

        masks = self.model.segmentation_head(decoder_output)

        return masks, extras

