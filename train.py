# train.py
import torch
import os
import cv2
import torch.nn as nn
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from thop import profile

from options import opt
from models.model import LiquidMamba
from utils.dataset import get_loader
from utils.func import clip_gradient, structure_loss
import py_sod_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
plt.switch_backend('agg')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def print_model_complexity(model, input_size, num_modalities):
    try:
        model.eval()
        dummy_inputs = [torch.randn(1, 3, input_size, input_size).cuda() for _ in range(num_modalities)]
        macs, params = profile(model, inputs=(dummy_inputs,), verbose=False)
        print("\n" + "=" * 50)
        print(f"[{opt.task} Task] LiquidMamba ({opt.num_modalities} Modalities)")
        print(f"VMamba: {opt.vmamba_type.upper()} | ConvNeXt: {opt.convnext_type.upper()}")
        print(f"Params: {params / 1e6:.4f} M")
        print(f"MACs:   {macs / 1e9:.4f} G")
        print("=" * 50 + "\n")
        model.train()
    except Exception as e:
        print(f"[Warning] Complexity calculation skipped: {e}")

def prepare_inputs(batch_data):
    # 1. 基础模态 RGB
    inputs = [batch_data['rgb'].cuda(non_blocking=True)]
    
    # 2. 第二模态 (Depth / Thermal / Flow)
    if opt.num_modalities >= 2: 
        inputs.append(batch_data['m2'].cuda(non_blocking=True))
        
    # 3. 第三模态 (Thermal / Depth)
    if opt.num_modalities == 3: 
        inputs.append(batch_data['m3'].cuda(non_blocking=True))
        
    gt = batch_data['gt'].cuda(non_blocking=True)
    return inputs, gt

def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Total Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_sm'], 'm-o', label='S-measure')
    plt.title('Validation Performance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_curves.png'))
    plt.close()

def visualize_first_val_sample(model, val_loader, epoch, save_path):
    model.eval()
    viz_dir = os.path.join(save_path, 'viz_evolution')
    os.makedirs(viz_dir, exist_ok=True)
    try:
        batch = next(iter(val_loader))
    except StopIteration:
        return

    inputs, _ = prepare_inputs(batch)
    shape = batch['shape'].view(-1).tolist()
    orig_h, orig_w = (shape[0], shape[1]) if len(shape) >= 2 else (512, 512)

    with torch.no_grad():
        preds = model(inputs)
        layer_names = ['Final', 'L1_Detail', 'L2_Mid', 'L3_Deep', 'L4_Semantic']
        
        for i, pred_logit in enumerate(preds):
            pred = torch.sigmoid(pred_logit)
            pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            res = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
            fname = batch['name'][0].split('.')[0]
            cv2.imwrite(os.path.join(viz_dir, f'ep{epoch:03d}_{layer_names[i]}_{fname}.png'), res)

def validate(model, val_loader, epoch, save_path, best_sm):
    model.eval()
    SM = py_sod_metrics.Smeasure()
    if val_loader is None: return 0.0, best_sm

    loader_iter = tqdm(val_loader, desc=f"  [Val] Ep {epoch}", leave=False, ncols=100)
    with torch.no_grad():
        for batch in loader_iter:
            inputs, _ = prepare_inputs(batch)
            
            with torch.cuda.amp.autocast():
                preds = model(inputs)
                res = preds[0]

            shape = batch['shape'].view(-1).tolist()
            orig_h, orig_w = (shape[0], shape[1]) if len(shape) >= 2 else (512, 512)

            res = F.interpolate(res, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            res = torch.sigmoid(res).squeeze()
            res_np = (res.cpu().numpy() * 255).astype(np.uint8)
            
            # 读取原始GT进行最纯粹的评估
            gt_path = batch['gt_path'][0]
            gt_np = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            if gt_np is None: continue
            
            # 异常尺寸保护
            if res_np.shape != gt_np.shape:
                res_np = cv2.resize(res_np, (gt_np.shape[1], gt_np.shape[0]))
                
            SM.step(res_np, gt_np)

    sm = SM.get_results()['sm']
    
    if sm > best_sm:
        best_sm = sm
        # 兼容单卡与多卡的权重保存，确保向下兼容
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state_dict, os.path.join(save_path, 'Best_Sm.pth'))
        print(f'  >>> [Save] New Best S-measure: {best_sm:.4f}')

    return sm, best_sm

def group_weight(model, backbone_lr, head_lr, weight_decay):
    g_decay_bb, g_no_decay_bb, g_decay_hd, g_no_decay_hd = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        is_backbone = ('backbone_global' in name) or ('backbone_local' in name)
        no_decay = (param.ndim <= 1) or ("bias" in name) or ("norm" in name) or ("bn" in name)

        if is_backbone:
            g_no_decay_bb.append(param) if no_decay else g_decay_bb.append(param)
        else:
            g_no_decay_hd.append(param) if no_decay else g_decay_hd.append(param)

    return [
        dict(params=g_decay_bb, lr=backbone_lr, weight_decay=weight_decay),
        dict(params=g_no_decay_bb, lr=backbone_lr, weight_decay=0.0),
        dict(params=g_decay_hd, lr=head_lr, weight_decay=weight_decay),
        dict(params=g_no_decay_hd, lr=head_lr, weight_decay=0.0),
    ]

if __name__ == '__main__':
    set_seed(opt.seed)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    exp_name = f"LMamba_{opt.task}_{opt.vmamba_type}_{opt.convnext_type}_{timestamp}"
    save_path = os.path.join(opt.save_path, exp_name)
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_path, 'log.txt'), level=logging.INFO)
    writer = SummaryWriter(log_dir=os.path.join(save_path, 'runs'))

    model = LiquidMamba(channel=64, 
                        convnext_type=opt.convnext_type, convnext_root=opt.convnext_pretrain,
                        vmamba_type=opt.vmamba_type, vmamba_root=opt.vmamba_pretrain,
                        num_modalities=opt.num_modalities).cuda()

    # 自动检测并启用多卡并行
    if torch.cuda.device_count() > 1:
        print(f">>> [Multi-GPU] Detected {torch.cuda.device_count()} GPUs. Using nn.DataParallel!")
        model = nn.DataParallel(model)

    # 打印复杂度时获取基础模型
    base_model = model.module if hasattr(model, 'module') else model
    print_model_complexity(base_model, opt.trainsize, opt.num_modalities)

    params_groups = group_weight(model, opt.lr * opt.backbone_lr_ratio, opt.lr, opt.weight_decay)
    optimizer = torch.optim.AdamW(params_groups, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler()

    train_loader = get_loader(opt.train_data_root, opt.train_datasets, opt.task, opt.batchsize, opt.trainsize, mode='train', num_workers=opt.num_workers)
    val_loader = get_loader(opt.val_data_root, opt.val_datasets, opt.task, 1, opt.testsize, mode='test', num_workers=4)

    warmup_epochs = 5
    lr_lambda = lambda ep: (ep + 1) / warmup_epochs if ep < warmup_epochs else 0.5 * (1 + np.cos(np.pi * (ep - warmup_epochs) / (opt.epoch - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_sm = 0.0
    history = {'train_loss': [], 'val_sm': []}

    print(f">>> Start Training ({opt.epoch} Epochs)...")
    for epoch in range(1, opt.epoch + 1):
        model.train()
        ep_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{opt.epoch}", ncols=100)

        for i, batch in enumerate(pbar, start=1):
            inputs, gts = prepare_inputs(batch)

            with torch.cuda.amp.autocast():
                preds = model(inputs)
                loss_final = structure_loss(preds[0], gts)
                loss_1 = structure_loss(preds[1], gts)
                loss_2 = structure_loss(preds[2], gts)
                loss_3 = structure_loss(preds[3], gts)
                loss_4 = structure_loss(preds[4], gts)
                
                loss = (loss_final + loss_1 + loss_2 + loss_3 + loss_4) / opt.accumulation_steps

            scaler.scale(loss).backward()
            ep_loss += loss.item() * opt.accumulation_steps

            if i % opt.accumulation_steps == 0 or i == len(train_loader):
                scaler.unscale_(optimizer)
                clip_gradient(optimizer, opt.clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix({'Loss': f"{loss.item() * opt.accumulation_steps:.3f}"})

        scheduler.step()
        avg_loss = ep_loss / len(train_loader)
        
        curr_sm, best_sm = validate(model, val_loader, epoch, save_path, best_sm)
        visualize_first_val_sample(model, val_loader, epoch, save_path)
        
        history['train_loss'].append(avg_loss)
        history['val_sm'].append(curr_sm)
        
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Metric/Smeasure', curr_sm, epoch)
        
        print(f"Ep {epoch} | Loss: {avg_loss:.4f} | SM: {curr_sm:.4f} (Best: {best_sm:.4f})")
        logging.info(f"Ep {epoch} | Loss: {avg_loss:.4f} | SM: {curr_sm:.4f}")
        
        plot_curves(history, save_path)

    writer.close()
    
    # 兼容单卡与多卡的权重保存，确保向下兼容
    final_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(final_state_dict, os.path.join(save_path, 'Final.pth'))
    
    print(f">>> Finished. Global Best S-measure: {best_sm:.4f}")