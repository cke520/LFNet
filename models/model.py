# models/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

try:
    import timm
except ImportError:
    print("[Error] 'timm' library is missing.")
    exit(1)

from models.adapter import VMambaAdapter
from models.layers import BasicConv2d, Liquid_Fusion_Cell, SaliencyGuidedUpsample, FinalRefinement

class ConvNeXtV2Backbone(nn.Module):
    def __init__(self, model_type='nano', pretrained_root=None):
        super().__init__()
        model_name = f'convnextv2_{model_type}'
        print(f">>> ConvNeXt (Local Shared): {model_name}")
        self.backbone = timm.create_model(model_name, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3))
        if pretrained_root:
            ckpt_path = os.path.join(pretrained_root, f"{model_name}_1k_224_ema.pt")
            if os.path.exists(ckpt_path):
                print(f"    [Load] ConvNeXt Weights: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
                self.backbone.load_state_dict(state_dict, strict=False)
        self.dims = self.backbone.feature_info.channels()

    def forward(self, x):
        return self.backbone(x)

class InnovativeModalityFusion(nn.Module):
    def __init__(self, in_dim, out_dim=64, num_modalities=1):
        super(InnovativeModalityFusion, self).__init__()
        self.num_modalities = num_modalities
        
        # ==========================================
        # [Unchanged] 单/双模态绝对兼容组件 -> 严禁修改
        # ==========================================
        # Base RGB projection
        self.proj_rgb = BasicConv2d(in_dim, out_dim, 1)
        
        if num_modalities >= 2:
            self.proj_m2 = BasicConv2d(in_dim, out_dim, 1)
            self.liquid = Liquid_Fusion_Cell(in_dim_h=out_dim, in_dim_l=out_dim, out_dim=out_dim)
            
        # ==========================================
        # [Restored] 三模态组件 -> 恢复为原VDT结构配置
        # (保持了你当前的变量名以防权重读取报错，但后续计算逻辑已改回串联)
        # ==========================================
        if num_modalities == 3:
            # 对应原代码中的 proj_v
            self.proj_tertiary = BasicConv2d(in_dim, out_dim, 1)
            
            # 对应原代码中的 lfc_temporal (将用作第二阶段串行演化)
            self.liquid_tertiary = Liquid_Fusion_Cell(in_dim_h=out_dim, in_dim_l=out_dim, out_dim=out_dim)
            

    def forward(self, feats):
        # 1. Extract primary cognitive state (RGB Liquid State)
        f_rgb = self.proj_rgb(feats[0])
        
        # ==========================================
        # [Single Modality] -> 绝对不可修改
        # ==========================================
        if self.num_modalities == 1:
            return f_rgb
            
        # ==========================================
        # [Dual Modality] -> 绝对不可修改
        # ==========================================
        elif self.num_modalities == 2:
            f_m2 = self.proj_m2(feats[1])
            fused = self.liquid(h=f_rgb, l=f_m2)
            return fused
            
        # ==========================================
        # [Tri Modality] -> 恢复为原本的VDT级联/串行演化机制
        # ==========================================
        elif self.num_modalities == 3:
            
            f_m2 = self.proj_m2(feats[1])
            f_tertiary = self.proj_tertiary(feats[2])
            
            feat_rd = self.liquid(h=f_rgb, l=f_m2)
            
            feat_final = self.liquid_tertiary(h=feat_rd, l=f_tertiary)
            
            return feat_final
        


class LiquidMamba(nn.Module):
    def __init__(self, channel=64, convnext_type='nano', vmamba_type='small', 
                 convnext_root=None, vmamba_root=None, freeze_backbone=False, num_modalities=1):
        super(LiquidMamba, self).__init__()
        self.num_modalities = num_modalities

        # --- 1. Siamese Backbones ---
        self.backbone_global = VMambaAdapter(model_type=vmamba_type, pretrained_root=vmamba_root)
        self.backbone_local = ConvNeXtV2Backbone(model_type=convnext_type, pretrained_root=convnext_root)

        self.g_dims = self.backbone_global.dims
        self.l_dims = self.backbone_local.dims

        if freeze_backbone:
            for param in self.backbone_global.parameters():
                param.requires_grad = False

        # --- 2. 创新模态融合 ---
        self.global_fusions = nn.ModuleList([
            InnovativeModalityFusion(self.g_dims[i], channel, num_modalities) for i in range(4)
        ])
        self.local_fusions = nn.ModuleList([
            InnovativeModalityFusion(self.l_dims[i], channel, num_modalities) for i in range(4)
        ])

        # --- 3. Liquid Decoder ---
        self.cell4 = Liquid_Fusion_Cell(in_dim_h=channel, in_dim_l=channel, out_dim=channel)
        self.cell3 = Liquid_Fusion_Cell(in_dim_h=channel * 2, in_dim_l=channel, out_dim=channel)
        self.cell2 = Liquid_Fusion_Cell(in_dim_h=channel * 2, in_dim_l=channel, out_dim=channel)
        self.cell1 = Liquid_Fusion_Cell(in_dim_h=channel * 2, in_dim_l=channel, out_dim=channel)

        self.up_3 = SaliencyGuidedUpsample(in_channels=channel, scale=2)
        self.up_2 = SaliencyGuidedUpsample(in_channels=channel, scale=2)
        self.up_1 = SaliencyGuidedUpsample(in_channels=channel, scale=2)

        # --- 4. Heads ---
        self.final_refine = FinalRefinement(channel)
        self.head4 = nn.Conv2d(channel, 1, 1)
        self.head3 = nn.Conv2d(channel, 1, 1)
        self.head2 = nn.Conv2d(channel, 1, 1)
        self.head1 = nn.Conv2d(channel, 1, 1)
        self.out_head = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Dropout2d(0.1),
            BasicConv2d(channel, channel // 2, 3, padding=1),
            nn.Conv2d(channel // 2, 1, 1)
        )
        self._init_module_weights()

    def _init_module_weights(self):
        for name, m in self.named_modules():
            if "backbone" in name or isinstance(m, Liquid_Fusion_Cell) or isinstance(m, InnovativeModalityFusion): 
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        size = inputs[0].size()[2:]

        # 1. Shared Feature Extraction
        g_feats_all = [self.backbone_global(x) for x in inputs]
        l_feats_all = [self.backbone_local(x) for x in inputs]

        # 2. Modality Fusion
        g_unified, l_unified = [], []
        for i in range(4):
            g_f = [f[i] for f in g_feats_all]
            l_f = [f[i] for f in l_feats_all]
            g_unified.append(self.global_fusions[i](g_f))
            l_unified.append(self.local_fusions[i](l_f))

        # 3. Liquid Decoder 
        # Stage 4
        g4, l4 = g_unified[3], l_unified[3]
        if l4.size()[2:] != g4.size()[2:]:
            l4 = F.interpolate(l4, size=g4.size()[2:], mode='bilinear', align_corners=False)
        h4 = self.cell4(h=g4, l=l4)
        out4 = self.head4(h4)

        # Stage 3
        h4_up = self.up_3(h4)
        g3 = g_unified[2]
        if h4_up.size()[2:] != g3.size()[2:]:
            h4_up = F.interpolate(h4_up, size=g3.size()[2:], mode='bilinear', align_corners=False)
        h3 = self.cell3(h=torch.cat([g3, h4_up], dim=1), l=l_unified[2])
        out3 = self.head3(h3)

        # Stage 2
        h3_up = self.up_2(h3)
        g2 = g_unified[1]
        if h3_up.size()[2:] != g2.size()[2:]:
            h3_up = F.interpolate(h3_up, size=g2.size()[2:], mode='bilinear', align_corners=False)
        h2 = self.cell2(h=torch.cat([g2, h3_up], dim=1), l=l_unified[1])
        out2 = self.head2(h2)

        # Stage 1
        h2_up = self.up_1(h2)
        g1 = g_unified[0]
        if h2_up.size()[2:] != g1.size()[2:]:
            h2_up = F.interpolate(h2_up, size=g1.size()[2:], mode='bilinear', align_corners=False)
        h1 = self.cell1(h=torch.cat([g1, h2_up], dim=1), l=l_unified[0])
        out1 = self.head1(h1)

        # 4. Final Output
        out_final = self.out_head(self.final_refine(h1))
        outs = [out_final, out1, out2, out3, out4]
        return [F.interpolate(o, size=size, mode='bilinear', align_corners=False) for o in outs]