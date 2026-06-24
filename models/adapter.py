import torch
import torch.nn as nn
import os

# 确保 models.encoders.vmamba 路径正确
from models.encoders.vmamba import Backbone_VSSM

class VMambaAdapter(nn.Module):
    """
    Global Stream: VMamba Backbone 动态配置 (Tiny / Small / Base)
    """
    def __init__(self, model_type='small', pretrained_root=None):
        super().__init__()
        
        # 严格映射各版本参数与对应的权重文件名
        if model_type == 'tiny':
            depths, dims, dp_rate = [2, 2, 9, 2], 96, 0.2
            ckpt_name = 'vssmtiny_dp01_ckpt_epoch_292.pth'
        elif model_type == 'small':
            depths, dims, dp_rate = [2, 2, 27, 2], 96, 0.3
            ckpt_name = 'vssmsmall_dp03_ckpt_epoch_238.pth'
        elif model_type == 'base':
            depths, dims, dp_rate = [2, 2, 27, 2], 128, 0.6
            ckpt_name = 'vssmbase_dp06_ckpt_epoch_241.pth'
        else:
            raise ValueError(f"Unsupported VMamba type: {model_type}")

        print(f">>> VMamba (Global Shared): {model_type.upper()} | Dims: {dims} | Depths: {depths}")

        self.vssm = Backbone_VSSM(
            num_classes=1000,
            depths=depths,
            dims=dims,
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=dp_rate
        )
        
        # 设置输出维度供给下游
        self.dims = [dims, dims * 2, dims * 4, dims * 8]

        # 自动拼接路径并加载
        if pretrained_root:
            ckpt_path = os.path.join(pretrained_root, ckpt_name)
            self.load_weights(ckpt_path)

    def load_weights(self, path):
        if not os.path.exists(path):
            print(f"    [Warning] VMamba weight file not found: {path}")
            return

        print(f"    [Load] VMamba Global Weights from: {path}")
        try:
            ckpt = torch.load(path, map_location='cpu')
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            new_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_dict[k[9:]] = v
                elif k.startswith('head') or k.startswith('classifier'):
                    continue
                else:
                    new_dict[k] = v
            msg = self.vssm.load_state_dict(new_dict, strict=False)
            print(f"    [Success] VMamba Weights loaded. Missing: {len(msg.missing_keys)}")
        except Exception as e:
            print(f"    [Error] VMamba Weight loading failed: {e}")

    def forward(self, x):
        return self.vssm(x)