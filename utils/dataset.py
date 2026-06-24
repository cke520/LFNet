# utils/dataset.py
import os
import torch
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

class UnifiedDataset(data.Dataset):
    def __init__(self, root, datasets, task, mode='train', trainsize=512, augment=True):
        self.trainsize = trainsize
        self.augment = augment
        self.mode = mode
        self.task = task
        self.datas = []

        print(f"\n" + "="*50)
        print(f"[*] Scanning '{mode}' datasets for Task: {task}")
        print(f"[*] Root Directory: {os.path.abspath(root)}")
        print("="*50)

        # 👑 严格规定每种任务需要读取 TXT 的最小列数
        if task == 'SOD':
            self.expected_cols = 2
        elif task in ['RGBD', 'RGBT', 'VSOD']:
            self.expected_cols = 3
        elif task in ['DVSOD', 'VDT']:
            self.expected_cols = 4
        else:
            raise ValueError(f"[Fatal Error] Unsupported task type: {task}")

        # 👑 [联合训练支持] 定义纯RGB数据集白名单。当遇到这些数据集时，自动忽略缺少深度/光流的报错
        self.pure_rgb_datasets = ['DUTS-TR', 'DUTS-TE', 'DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S']

        for dataset in datasets:
            dataset_sample_count = 0
            
            # 👑 [联合训练支持] 判断当前数据集是否为外挂的纯RGB数据集
            is_pure_rgb_dataset = any(name in dataset for name in self.pure_rgb_datasets)
            
            data_dir = os.path.join(root, dataset)
            txt_path = os.path.join(data_dir, f'{mode}.txt')

            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"\n[🚨 Fatal Error] Dataset directory not found: {data_dir}")

            # =================================================================
            # 策略 1: 优先尝试读取 TXT 文件 (标准流程)
            # =================================================================
            if os.path.exists(txt_path):
                print(f"  -> [OK] Reading TXT file: {txt_path}")
                if is_pure_rgb_dataset:
                    print(f"     [Info] Identified as Pure RGB auxiliary dataset. Relaxing column checks.")

                with open(txt_path, 'r') as f:
                    lines = f.readlines()
                    for line_idx, line in enumerate(lines):
                        parts = line.strip().split()
                        if not parts: continue
                        
                        # 👑 [联合训练支持] 如果是联合训练的RGB数据集，只要求有2列即可
                        current_expected_cols = 2 if is_pure_rgb_dataset else self.expected_cols

                        # 列数检查
                        if len(parts) < current_expected_cols:
                            raise ValueError(
                                f"\n[🚨 Fatal Error] TXT Format Error in {txt_path} (Line {line_idx + 1})\n"
                                f"Task '{task}' requires AT LEAST {current_expected_cols} columns, but found {len(parts)}.\n"
                                f"Line content: {line.strip()}"
                            )

                        item = {'dataset': dataset}
                        
                        # 解析路径
                        item['rgb'] = self._resolve_path(data_dir, root, parts[0], "RGB", line_idx+1)
                        item['gt'] = self._resolve_path(data_dir, root, parts[1], "GT", line_idx+1)
                        
                        # 👑 [联合训练支持] 对于联合训练数据，不强求读取 m2 和 m3
                        if len(parts) >= 3 and not is_pure_rgb_dataset:
                            item['m2'] = self._resolve_path(data_dir, root, parts[2], "M2", line_idx+1)
                        if len(parts) >= 4 and not is_pure_rgb_dataset:
                            item['m3'] = self._resolve_path(data_dir, root, parts[3], "M3", line_idx+1)
                        
                        self.datas.append(item)
                        dataset_sample_count += 1
            
            # =================================================================
            # 策略 2: 如果 TXT 不存在，进入通用文件夹扫描模式 (核心修复)
            # =================================================================
            else:
                print(f"  -> [Info] '{mode}.txt' missing in {dataset}. Entering Auto-Scan Mode.")
                
                # 定义常用文件夹别名
                rgb_names = ['V', 'RGB', 'Imgs', 'image', 'images', 'Left'] # VSOD有时用Left
                gt_names = ['GT', 'GroundTruth', 'gt', 'masks']
                depth_names = ['D', 'Depth', 'depth', 'depths']
                thermal_names = ['T', 'Thermal', 'Infrared', 'thermal', 'infra']

                # 辅助函数：查找子文件夹
                def find_subdir(base, candidates):
                    for c in candidates:
                        p = os.path.join(base, c)
                        if os.path.isdir(p): return p
                    return None
                
                # 辅助函数：根据前缀查找文件
                def get_file_path(folder, prefix):
                    if not folder: return None
                    for ext in ['.png', '.jpg', '.bmp', '.jpeg', '.tif']:
                        path = os.path.join(folder, prefix + ext)
                        if os.path.exists(path): return path
                    return None

                # 1. 基础模态 (所有任务都需要)
                p_rgb = find_subdir(data_dir, rgb_names)
                p_gt = find_subdir(data_dir, gt_names)

                if p_rgb and p_gt:
                    # 2. 动态决定还需要找什么文件夹
                    p_m2_target = None # 对应 m2 的文件夹
                    p_m3_target = None # 对应 m3 的文件夹
                    
                    # 👑 [联合训练支持] 如果是联合训练纯RGB数据集，直接无视 m2 和 m3 的文件夹匹配
                    if not is_pure_rgb_dataset:
                        # RGB-D / RGB-T / VDT 的逻辑分流
                        if task == 'RGBD':
                            p_m2_target = find_subdir(data_dir, depth_names) # m2 = Depth
                            if not p_m2_target: print(f"     [Warning] RGBD task but 'Depth' folder not found in {dataset}.")
                        
                        elif task == 'RGBT':
                            p_m2_target = find_subdir(data_dir, thermal_names) # m2 = Thermal
                            if not p_m2_target: print(f"     [Warning] RGBT task but 'Thermal' folder not found in {dataset}.")

                        elif task == 'VDT':
                            p_m2_target = find_subdir(data_dir, depth_names)   # m2 = Depth
                            p_m3_target = find_subdir(data_dir, thermal_names) # m3 = Thermal
                            if not (p_m2_target and p_m3_target): 
                                print(f"     [Warning] VDT task but 'Depth' or 'Thermal' folder missing in {dataset}.")

                    # VSOD 特殊处理 (通常很难自动扫描Flow，只扫描RGB+GT作为基础，或者如果有Depth/Flow文件夹)
                    # 这里保持基础鲁棒性，如果有 TXT 会走上面，没 TXT 至少能跑 RGB+GT

                    # 3. 遍历图片进行匹配
                    images = sorted([x for x in os.listdir(p_rgb) if x.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))])
                    
                    for img_name in images:
                        name_prefix = img_name.rsplit('.', 1)[0]
                        
                        f_rgb = os.path.join(p_rgb, img_name)
                        f_gt = get_file_path(p_gt, name_prefix)
                        
                        # 必须有 RGB 和 GT
                        if f_rgb and f_gt:
                            item = {
                                'dataset': dataset,
                                'rgb': f_rgb,
                                'gt': f_gt
                            }
                            
                            is_valid_sample = True
                            
                            # 尝试匹配 m2
                            if p_m2_target:
                                f_m2 = get_file_path(p_m2_target, name_prefix)
                                if f_m2: item['m2'] = f_m2
                                else: is_valid_sample = False # 缺模态则丢弃
                            
                            # 尝试匹配 m3
                            if p_m3_target:
                                f_m3 = get_file_path(p_m3_target, name_prefix)
                                if f_m3: item['m3'] = f_m3
                                else: is_valid_sample = False # 缺模态则丢弃

                            # 如果所有需要的模态都找到了
                            if is_valid_sample:
                                self.datas.append(item)
                                dataset_sample_count += 1
                else:
                     # 连 RGB 或 GT 文件夹都找不到
                     pass # 将在后面 dataset_sample_count == 0 时统一报错

            # 👑 核心修复：如果这个数据集一张图都没加载进来，直接崩溃报错！
            if dataset_sample_count == 0:
                raise RuntimeError(
                    f"\n[🚨 Fatal Error] Dataset '{dataset}' yielded 0 valid samples!\n"
                    f"Reason: Missing '{mode}.txt' AND failed to auto-match folders (RGB/GT/Depth/Thermal).\n"
                    f"Check path: {data_dir}"
                )
            else:
                print(f"     => Successfully loaded {dataset_sample_count} samples from '{dataset}'.")

        print("="*50)
        print(f"[*] Total loaded samples across all datasets: {len(self.datas)}")
        print("="*50 + "\n")
        
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)

    def _resolve_path(self, data_dir, root_dir, rel_path, modality_name="File", line_idx=0):
        """完全还原原本 VSOD 代码中正确的寻址逻辑"""
        if not rel_path: 
            raise ValueError(f"[🚨 Fatal Error] Path for {modality_name} is empty in TXT file (Line {line_idx})!")
        
        # 1. 还原原版有效逻辑：直接字符串拼接！
        cand1 = data_dir + rel_path
        if os.path.exists(cand1): return cand1
        
        # 2. 常规 os.path.join (剥离前缀斜杠)
        clean_path = rel_path.lstrip('/\\')
        cand2 = os.path.join(data_dir, clean_path)
        if os.path.exists(cand2): return cand2
        
        # 3. 如果 TXT 里的路径包含了 Dataset 名字，退回 root_dir 拼接
        cand3 = os.path.join(root_dir, clean_path)
        if os.path.exists(cand3): return cand3
        
        # 👑 遵从指令：找不到文件直接报错
        raise FileNotFoundError(
            f"\n\n[🚨 Fatal Error] {modality_name} File Missing on Disk!\n"
            f"  Error Source : TXT Line {line_idx}\n"
            f"  TXT String   : '{rel_path}'\n"
            f"  Tried looking in:\n"
            f"    1. {cand1}\n"
            f"    2. {cand2}\n"
            f"    3. {cand3}\n"
            f"  Please make sure the file actually exists!"
        )

    def _load_img(self, path, mode='RGB'):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"[🚨 Fatal Error] Attempted to load an invalid or missing image path: '{path}'")
        try:
            return Image.open(path).convert(mode)
        except Exception as e:
            raise RuntimeError(f"[🚨 Fatal Error] Corrupted image file: {path} | Details: {e}")

    def __getitem__(self, index):
        item = self.datas[index]
        
        rgb = self._load_img(item['rgb'], 'RGB')
        gt = self._load_img(item['gt'], 'L')

        # 👑 光流图(Flow)必须是 RGB 以保留方向信息！深度图(Depth)是灰度 L
        if self.task in ['VSOD', 'DVSOD', 'VDT'] and 'm2' in item:
            # VDT: m2 是 Depth, Unified 逻辑通常将其视作 RGB 读取 (兼容旧 VDTDataset)
            # RGBD: m2 是 Depth
            m2 = self._load_img(item['m2'], 'RGB')
            is_rgb_flow = True
        elif 'm2' in item:
            # RGBT: m2 是 Thermal, 通常 L 读取
            m2 = self._load_img(item['m2'], 'L') 
            is_rgb_flow = False
        else:
            m2 = None
            is_rgb_flow = False

        # 👑 动态判断 m3 (Depth/Thermal) 的读取模式
        if 'm3' in item:
            if self.task == 'DVSOD':
                m3 = self._load_img(item['m3'], 'RGB')
                is_rgb_m3 = True
            else:
                # VDT: m3 是 Thermal
                m3 = self._load_img(item['m3'], 'L')
                is_rgb_m3 = False
        else:
            m3 = None
            is_rgb_m3 = False

        orig_w, orig_h = gt.size

        if self.augment and self.mode == 'train':
            rgb = rgb.resize((self.trainsize, self.trainsize), Image.BILINEAR)
            gt = gt.resize((self.trainsize, self.trainsize), Image.NEAREST)
            if m2: m2 = m2.resize((self.trainsize, self.trainsize), Image.BILINEAR)
            if m3: m3 = m3.resize((self.trainsize, self.trainsize), Image.BILINEAR)

            if random.random() < 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
                if m2: m2 = m2.transpose(Image.FLIP_LEFT_RIGHT)
                if m3: m3 = m3.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() < 0.5:
                angle = random.randint(-10, 10)
                rgb = rgb.rotate(angle, resample=Image.BILINEAR)
                gt = gt.rotate(angle, resample=Image.NEAREST)
                if m2: m2 = m2.rotate(angle, resample=Image.BILINEAR)
                if m3: m3 = m3.rotate(angle, resample=Image.BILINEAR)

            if random.random() < 0.5:
                rgb = self.color_jitter(rgb)
        else:
            rgb = rgb.resize((self.trainsize, self.trainsize), Image.BILINEAR)
            gt = gt.resize((self.trainsize, self.trainsize), Image.NEAREST)
            if m2: m2 = m2.resize((self.trainsize, self.trainsize), Image.BILINEAR)
            if m3: m3 = m3.resize((self.trainsize, self.trainsize), Image.BILINEAR)

        # 转换为 Tensor
        arr_rgb = np.array(rgb).astype(np.float32) / 255.0
        arr_rgb = (arr_rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor_rgb = torch.from_numpy(arr_rgb).permute(2, 0, 1).float()
        
        arr_gt = np.array(gt).astype(np.float32) / 255.0
        tensor_gt = torch.from_numpy(arr_gt).unsqueeze(0).float().ge(0.5).float()

        path_parts = item['rgb'].replace('\\', '/').split('/')
        vid_name = path_parts[-3] if len(path_parts) >= 3 else (path_parts[-2] if len(path_parts) >= 2 else 'unknown')

        out_dict = {
            'rgb': tensor_rgb,
            'gt': tensor_gt,
            'gt_path': item['gt'],  
            'name': path_parts[-1], 
            'dataset': item['dataset'],
            'shape': torch.tensor([orig_h, orig_w]),
            'vid_name': vid_name 
        }

        # 动态处理辅助模态
        def to_tensor_aux(img, is_rgb=False):
            arr = np.array(img).astype(np.float32) / 255.0
            if not is_rgb:
                arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2) 
            arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            return torch.from_numpy(arr).permute(2, 0, 1).float()

        # 👑 [联合训练支持] 如果当前任务期望多模态，但数据缺失（比如DUTS-TR），则填补全零Tensor！
        if self.task in ['RGBD', 'RGBT', 'VSOD', 'DVSOD', 'VDT']:
            if m2 is not None:
                out_dict['m2'] = to_tensor_aux(m2, is_rgb=is_rgb_flow)
            else:
                out_dict['m2'] = torch.zeros((3, self.trainsize, self.trainsize))

        if self.task in ['DVSOD', 'VDT']:
            if m3 is not None:
                out_dict['m3'] = to_tensor_aux(m3, is_rgb=is_rgb_m3)
            else:
                out_dict['m3'] = torch.zeros((3, self.trainsize, self.trainsize))

        return out_dict

    def __len__(self):
        return len(self.datas)

def get_loader(root, datasets, task, batchsize, trainsize, mode='train', num_workers=4):
    dataset = UnifiedDataset(root, datasets, task, mode=mode, trainsize=trainsize, augment=(mode == 'train'))
    return data.DataLoader(dataset, batch_size=batchsize, shuffle=(mode == 'train'),
                           num_workers=num_workers, pin_memory=True, drop_last=(mode == 'train'))