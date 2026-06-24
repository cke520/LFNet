# options.py
import argparse
import os

parser = argparse.ArgumentParser()

# --- 1. Task Config ---
parser.add_argument('--task', type=str, default='SOD', choices=['SOD', 'RGBD', 'RGBT', 'VSOD', 'DVSOD', 'VDT'])

# --- 2. Training Config ---
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--backbone_lr_ratio', type=float, default=0.05)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--batchsize', type=int, default=2)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--trainsize', type=int, default=512)
parser.add_argument('--testsize', type=int, default=512)
parser.add_argument('--clip', type=float, default=0.5)
parser.add_argument('--gpu_id', type=str, default='0')

# --- 3. Model Config ---
parser.add_argument('--convnext_type', type=str, default='pico', choices=['atto', 'pico', 'nano'])
parser.add_argument('--convnext_pretrain', type=str, default='../pretrain/')

parser.add_argument('--vmamba_type', type=str, default='small', choices=['tiny', 'small', 'base'])
parser.add_argument('--vmamba_pretrain', type=str, default='../pretrain/')

parser.add_argument('--freeze_backbone', action='store_true')
parser.add_argument('--save_path', type=str, default='./ckpt/LiquidMamba_Unified/')
parser.add_argument('--test_model_path', type=str, default='', help='Specific ckpt path for testing')

# --- 4. Dataset Config ---
parser.add_argument('--train_data_root', type=str, default='../Data/RGB/train/')
parser.add_argument('--val_data_root', type=str, default='../Data/RGB/test/')

parser.add_argument('--train_datasets', nargs='+', default=[''], help='Datasets for training')
# 验证集通常选一个较小的测试集，如 PASCAL-S, NLPR, DAVIS
parser.add_argument('--val_datasets', nargs='+', default=['PASCAL-S'], help='Dataset for validation during training')

opt = parser.parse_args()

# 自动推导模态数量
if opt.task == 'SOD':
    opt.num_modalities = 1
elif opt.task in ['RGBD', 'RGBT', 'VSOD']:
    opt.num_modalities = 2
elif opt.task in ['DVSOD', 'VDT']:
    opt.num_modalities = 3
else:
    raise ValueError(f"Unknown task: {opt.task}")