import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
from tqdm import tqdm
from options import opt
from models.model import LiquidMamba
from utils.dataset import get_loader

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# ==============================================================
# 👑 核心配置：内部管理各任务的专属测试集
# ==============================================================
TASK_TEST_CONFIGS = {
    'SOD': {
        'root': '../Data/SOD/test/',
        'datasets': ['DUTS-TE', 'ECSSD', 'HKU-IS', 'PASCAL-S', 'DUT-OMRON']
    },
    'RGBD': {
        'root': '../Data/RGBD/test/',
        'datasets': ['NJU2K', 'NLPR', 'SIP', 'STERE', 'DUT-RGBD']
    },
    'RGBT': {
        'root': '../Data/RGBT/test/',
        'datasets': ['VT821', 'VT1000', 'VT5000']
    },
    'VSOD': {
        'root': '../Data/VSOD/test/',
        'datasets': ['DAVIS', 'DAVSOD', 'FBMS', 'SegTrack-V2', 'VOS']
    },
    'DVSOD': {
        'root': '../Data/DVSOD/',
        'datasets': ['RDVS']
    },
    'VDT': {
        'root': '../Data/VDT/test/',
        'datasets': ['VDT-2048']
    }
}

def test():
    print("\n" + "=" * 50)
    print(f">>> Testing Task: {opt.task} | Modalities: {opt.num_modalities}")
    print("=" * 50 + "\n")
    
    # 1. 初始化模型
    model = LiquidMamba(channel=64, 
                        convnext_type=opt.convnext_type, 
                        vmamba_type=opt.vmamba_type,
                        num_modalities=opt.num_modalities).cuda()
    
    # 2. 查找最佳权重
    ckpt_path = opt.test_model_path
    if not os.path.exists(ckpt_path):
        subdirs = [os.path.join(opt.save_path, d) for d in os.listdir(opt.save_path) if os.path.isdir(os.path.join(opt.save_path, d))]
        if subdirs:
            latest_dir = max(subdirs, key=os.path.getmtime)
            ckpt_path = os.path.join(latest_dir, 'Best_Sm.pth')
            
    if not os.path.exists(ckpt_path):
        print(f"[Error] Checkpoint missing: {ckpt_path}")
        return
        
    print(f"    [Load] Weights from: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location='cuda'), strict=False)
    model.eval()

    # 以任务名称作为根目录
    save_root = os.path.join('./results', opt.task)

    # 3. 读取该任务的专属测试配置
    test_config = TASK_TEST_CONFIGS[opt.task]
    test_root = test_config['root']
    test_datasets = test_config['datasets']

    # 4. 循环遍历所有测试集
    for dataset_name in test_datasets:
        print(f"\nProcessing {dataset_name}...")
        
        # 验证测试集目录是否存在
        dataset_dir = os.path.join(test_root, dataset_name)
        if not os.path.exists(dataset_dir):
            print(f"[Warning] Dataset path not found: {dataset_dir}. Skipped.")
            continue
            
        test_loader = get_loader(test_root, [dataset_name], opt.task, 1, opt.testsize, mode='test', num_workers=4)
        
        with torch.no_grad():
            for batch in tqdm(test_loader, ncols=100):
                # ====================================================
                # 👑 核心修复 1：严格对齐 train.py 的输入模态顺序
                # ====================================================
                inputs = [batch['rgb'].cuda()]
                
                if opt.task == 'DVSOD' and opt.num_modalities == 3:
                    # DVSOD 专属修复：将 m3(Depth) 放在第二位，m2(Flow) 放在第三位
                    inputs.append(batch['m3'].cuda())
                    inputs.append(batch['m2'].cuda())
                else:
                    # 其余 5 个任务维持正常顺序
                    if opt.num_modalities >= 2: inputs.append(batch['m2'].cuda())
                    if opt.num_modalities == 3: inputs.append(batch['m3'].cuda())

                # 前向推理
                preds = model(inputs)
                res = preds[0] if isinstance(preds, list) else preds
                
                # 获取原图尺寸，保证评测时严格对齐
                shape = batch['shape'].view(-1).tolist()
                orig_h, orig_w = (shape[0], shape[1]) if len(shape) >= 2 else (512, 512)

                # 插值与归一化
                res = F.interpolate(res, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                res = (res * 255).astype(np.uint8)

                # 文件名处理
                f_name = os.path.splitext(batch['name'][0])[0] + '.png'
                
                # ====================================================
                # 👑 核心修复 2：精准控制视频/平铺数据集的保存结构
                # ====================================================
                # 仅 VSOD 和 DVSOD 存在内部的视频子文件夹
                if opt.task in ['VSOD', 'DVSOD']:
                    v_name = batch['vid_name'][0]
                    # 额外保护：如果提取出的视频名和数据集同名，或者找不到，强制平铺（以防个别特例）
                    if v_name == dataset_name or v_name == 'unknown':
                        save_dir = os.path.join(save_root, dataset_name)
                    else:
                        save_dir = os.path.join(save_root, dataset_name, v_name)
                else:
                    # 其他任务 (SOD, RGBD, RGBT, VDT) 全部是平铺的图片
                    # 例如 VDT-2048 将直接保存在 results/VDT/VDT-2048/ 目录下
                    save_dir = os.path.join(save_root, dataset_name)
                    
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f_name), res)

    print(f"\n>>> All Tests Finished. Results are saved in '{save_root}/'")

if __name__ == '__main__':
    test()