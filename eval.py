import os
import cv2
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import py_sod_metrics
import warnings

# 👑 核心修复：屏蔽掉 py_sod_metrics 烦人的旧版 Fmeasure 移除警告，保持终端干净
warnings.filterwarnings('ignore', category=UserWarning, module='py_sod_metrics')

# ==============================================================
# 👑 大一统评估配置：严格对应 test.py 中的目录结构
# ==============================================================
TASK_TEST_CONFIGS = {
    'SOD': {
        'root': '../Data/SOD/test/',
        'datasets': ['DUTS-TE', 'DUT-OMRON', 'HKU-IS', 'PASCAL-S', 'ECSSD']
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
        # 'datasets': ['RDVS']
        # 'datasets': ['DVisal']
        'datasets': ['vidsod_100']
    },
    'VDT': {
        'root': '../Data/VDT/test/',
        'datasets': ['VDT-2048']
    }
}

PRED_ROOT_BASE = './results/'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='DVSOD', choices=['SOD', 'RGBD', 'RGBT', 'VSOD', 'DVSOD', 'VDT'])
    return parser.parse_args()

def parse_vid_and_fname(img_rel_path):
    """
    👑 核心改进：与 dataset.py 中的视频名解析逻辑 100% 严格对齐！
    解决原本 os.path.dirname 会把 '/select_0043/Imgs/0001.png'
    的视频名错误解析为 'Imgs' 导致找不到预测图的致命 Bug！
    """
    img_rel_path = img_rel_path.replace('\\', '/')
    path_parts = img_rel_path.split('/')
    
    # 提取文件名
    fname = path_parts[-1]
    
    # 与 dataset.py 完全一致的提取逻辑：
    if len(path_parts) >= 3:
        vid_name = path_parts[-3]
    elif len(path_parts) >= 2:
        vid_name = path_parts[-2]
    else:
        vid_name = 'unknown'
        
    return vid_name, fname

def evaluate():
    opt = parse_args()
    config = TASK_TEST_CONFIGS[opt.task]
    data_root = config['root']
    datasets = config['datasets']

    pred_task_root = os.path.join(PRED_ROOT_BASE, opt.task)

    results_list = []
    print("\n" + "=" * 60)
    print(f">>> Start Evaluation for Task: {opt.task}")
    print(f"    Data Root: {os.path.abspath(data_root)}")
    print(f"    Pred Root: {os.path.abspath(pred_task_root)}")
    print("=" * 60 + "\n")

    for dname in datasets:
        ds_root = os.path.join(data_root, dname)
        pred_root = os.path.join(pred_task_root, dname)
        txt_path = os.path.join(ds_root, 'test.txt')

        if not os.path.exists(pred_root):
            print(f"[Skip] Prediction folder not found: {pred_root}")
            continue

        # 👑 核心修复：退回旧版 Fmeasure，保证返回值字典兼容你的 Excel 输出逻辑
        MAE = py_sod_metrics.MAE()
        SM = py_sod_metrics.Smeasure()
        FM = py_sod_metrics.Fmeasure()  
        EM = py_sod_metrics.Emeasure()

        valid_files = 0

        print(f"Evaluating {dname}...")

        # ==========================================
        # 模式 A: 基于 TXT 列表 (适用于 VSOD, DVSOD, VDT)
        # ==========================================
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            for line_idx, line in enumerate(tqdm(lines, desc=f"  [TXT Mode]", ncols=100, leave=False)):
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"[🚨 Fatal Error] TXT format error in {txt_path} (Line {line_idx+1}). Missing GT path.")
                
                img_rel, gt_rel = parts[0], parts[1]

                # 👑 核心改进：采用与 dataset.py 相同的直拼/容错寻址，防止 GT 找不到
                gt_path = None
                cand1 = ds_root + gt_rel
                clean_path = gt_rel.lstrip('/\\')
                cand2 = os.path.join(ds_root, clean_path)
                cand3 = os.path.join(data_root, clean_path)
                
                if os.path.exists(cand1): gt_path = cand1
                elif os.path.exists(cand2): gt_path = cand2
                elif os.path.exists(cand3): gt_path = cand3
                
                # 👑 零容忍：找不到 GT 直接报错，不跳过！
                if not gt_path:
                    raise FileNotFoundError(
                        f"\n[🚨 Fatal Error] Ground Truth File Missing during Evaluation!\n"
                        f"  Dataset: {dname}\n"
                        f"  TXT String: '{gt_rel}'\n"
                        f"  Please ensure your test dataset GT files actually exist!"
                    )

                # 2. 解析预测图路径 (使用修复后的解析器)
                vid_name, fname = parse_vid_and_fname(img_rel)
                fname_png = os.path.splitext(fname)[0] + '.png'
                
                # 兼容查找：视频子目录 OR 直接在根目录
                pred_path_vid = os.path.join(pred_root, vid_name, fname_png)
                pred_path_flat = os.path.join(pred_root, fname_png)
                
                if os.path.exists(pred_path_vid):
                    pred_path = pred_path_vid
                elif os.path.exists(pred_path_flat):
                    pred_path = pred_path_flat
                else:
                    # 👑 零容忍：测试集中要求的图，预测文件夹里竟然没有？直接报错！
                    raise FileNotFoundError(
                        f"\n[🚨 Fatal Error] Prediction Map Missing!\n"
                        f"  Missing file: {pred_path_vid}\n"
                        f"  Please make sure test.py processed all images for dataset '{dname}' successfully."
                    )

                # 3. 读取与指标计算
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                
                if gt is None or pred is None: 
                    raise RuntimeError(f"[🚨 Fatal Error] Failed to read image: GT or Pred is corrupted. Check {fname_png}")
                
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

                MAE.step(pred, gt)
                SM.step(pred, gt)
                FM.step(pred, gt)
                EM.step(pred, gt)
                valid_files += 1

        # ==========================================
        # 模式 B: 基于纯文件夹遍历 (容错能力极强，支持子文件夹递归)
        # ==========================================
        else:
            # 自动推断 GT 目录
            gt_dir = None
            for c in ['GT', 'masks', 'mask', 'ground-truth']:
                if os.path.exists(os.path.join(ds_root, c)):
                    gt_dir = os.path.join(ds_root, c)
                    break
            
            if not gt_dir:
                print(f"  [Warning] GT folder not found in {ds_root}. Skipped.")
                continue

            # 递归搜索预测目录，完美支持 VSOD 的多级子文件夹
            pred_files = []
            for r, d, f in os.walk(pred_root):
                for file in f:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # 记录相对于 pred_root 的相对路径 (例如 'bike/0000.png')
                        pred_files.append(os.path.relpath(os.path.join(r, file), pred_root))
            
            for rel_pred in tqdm(pred_files, desc=f"  [Folder Mode]", ncols=100, leave=False):
                pred_path = os.path.join(pred_root, rel_pred)
                base_no_ext = os.path.splitext(rel_pred)[0]
                
                # 智能匹配 GT 扩展名 (支持子文件夹自动对齐)
                gt_path = None
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    tmp = os.path.join(gt_dir, base_no_ext + ext)
                    if os.path.exists(tmp):
                        gt_path = tmp
                        break
                        
                if not gt_path: continue

                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

                if gt is None or pred is None: continue
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

                MAE.step(pred, gt)
                SM.step(pred, gt)
                FM.step(pred, gt)
                EM.step(pred, gt)
                valid_files += 1

        # ==========================================
        # 结果统计与打印
        # ==========================================
        if valid_files > 0:
            sm = SM.get_results()['sm']
            mae = MAE.get_results()['mae']
            maxF = FM.get_results()['fm']['curve'].max()
            maxE = EM.get_results()['em']['curve'].max()

            print(f"  [{dname:<10}] Images: {valid_files} | Sm: {sm:.4f} | maxF: {maxF:.4f} | maxE: {maxE:.4f} | MAE: {mae:.4f}")

            results_list.append({
                'Dataset': dname,
                'S-m': round(sm, 4),
                'maxF': round(maxF, 4),
                'maxE': round(maxE, 4),
                'MAE': round(mae, 4)
            })
        else:
            print(f"  [{dname:<10}] [Warning] No valid files evaluated.")

    # ==========================================
    # 保存至 Excel
    # ==========================================
    if results_list:
        df = pd.DataFrame(results_list)
        # 强制格式化列顺序为论文常用格式
        df = df[['Dataset', 'S-m', 'maxF', 'maxE', 'MAE']]

        save_file = f'LiquidMamba_{opt.task}_Evaluation.xlsx'
        df.to_excel(save_file, index=False)
        print("\n" + "=" * 60)
        print(f">>> Evaluation Complete! Excel Report saved to: {os.path.abspath(save_file)}")
        print("=" * 60)
    else:
        print("\n>>> No datasets evaluated.")

if __name__ == '__main__':
    evaluate()