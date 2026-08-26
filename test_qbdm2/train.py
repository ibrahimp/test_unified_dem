#!/usr/bin/env python3
"""
Lightweight Quad Bayer demosaic experiment with ISO 12800 support.
Updated: Linear Output Support + Symmetric Noise + Increased Gradient Clipping.
"""

import argparse
import math
import pickle
import random
from pathlib import Path
import pathlib

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import build_model, make_input_channels
from dataset import (
    list_images, unified_patch_key, load_hard_patch_keys, 
    resolve_hard_patches, list_unified_patches, QuadBayerDataset
)

torch.serialization.add_safe_globals([Path])
torch.serialization.add_safe_globals([pathlib.WindowsPath, pathlib.PosixPath])

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PATTERNS = {"grbg", "rggb", "gbrg", "bggr"}
COLOR_TO_CHANNEL = {"r": 0, "g": 1, "b": 2}
UNIFIED_SPLITS = {
    "train": [3, 4, 5, 6, 9, 10, 11, 12, 13, 14],
    "val": [1, 2], 
    "test": [7, 8, 15, 16, 17],
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight Quad Bayer/OCL demosaicer.")
    parser.add_argument("--data-root", required=True, help="Image-folder root or Unified patch dataset root.")
    parser.add_argument("--out-dir", default="runs/lightweight_qb", help="Output directory for checkpoints.")
    parser.add_argument("--dataset-mode", default="auto", choices=("auto", "folders", "unified_patches"))
    parser.add_argument("--hard_patches", default=None, help="Optional hardpatches*.pkl file for Unified train filtering.")
    parser.add_argument("--hard_patch_percentile", type=float, default=None, help="Convenience lookup for hardpatches{p:.2f}.pkl under data-root.")
    parser.add_argument("--npy_scale", type=float, default=13496.0, help="Scale factor for Unified uint16 linear RGB .npy files.")
    parser.add_argument("--max_train_images", type=int, default=0, help="Limit train samples for smoke tests.")
    parser.add_argument("--max_val_images", type=int, default=0, help="Limit val samples for smoke tests.")
    parser.add_argument("--pattern", default="grbg", choices=sorted(PATTERNS), help="Color pattern (e.g., 'rg' or 'grbg').")
    parser.add_argument("--mode", default="quad_bayer", choices=("bayer", "quad_bayer"), help="Sensor layout mode.")
    parser.add_argument("--multi_pattern", action="store_true", help="Randomly sample from all Quad Bayer patterns during training.")
    parser.add_argument("--patch-size", type=int, default=256, help="Training crop size.")
    parser.add_argument("--epochs", type=int, default=150) 
    parser.add_argument("--batch-size", type=int, default=32) 
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Initial learning rate.")
    parser.add_argument("--lr_end", type=float, default=1.0e-6, help="Final learning rate.")
    parser.add_argument("--width", type=int, default=80, help="Simplified-NAFBlock trunk width.")
    parser.add_argument("--blocks", type=int, default=64, help="Number of Simplified_NAFBlocks.")
    parser.add_argument("--downsample", type=int, default=4, choices=(1, 2, 3, 4), help="Initial spatial downsampling factor.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--use_cfa", action="store_true", help="Concatenate R/G/B one-hot CFA masks.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every X epochs.")
    
    # Luma-Chroma Loss Weights
    parser.add_argument("--weight_g", type=float, default=1.0, help="Weight for Green channel loss (Luminance).")
    parser.add_argument("--weight_rg", type=float, default=1.0, help="Weight for Red-Green difference loss.")
    parser.add_argument("--weight_bg", type=float, default=1.0, help="Weight for Blue-Green difference loss.")

    parser.add_argument("--resume", action="store_true", help="Resume training from best.pt")
    parser.add_argument("--save_val_visual", action="store_true", help="Save a validation sample for visualization.")
    parser.add_argument("--val_scene_name", type=str, default=None, help="Substring/filename to filter for visualization saving.")
    parser.add_argument("--gt_only", action="store_true", help="Train only on Ground Truth pairs.")
    parser.add_argument("--debug_film", action="store_true", help="Enable FiLM parameter debug logging (slows training).")
    
    # Extended Noise & Binning Arguments
    parser.add_argument("--max-iso", type=int, default=12800, help="Maximum ISO level for synthetic noise during training.")
    parser.add_argument("--binning", type=int, default=1, help="Binning factor (e.g., 2 for 2x2 binning).")
    
    # Pedestal argument completely removed
    
    parser.add_argument("--val-iso-levels", type=int, nargs='+', 
                        default=[400, 800, 3200, 6400, 12800], 
                        help="List of ISO levels to evaluate separately during validation.")
    
    parser.add_argument("--weight-decay", type=float, default=1.0e-3, help="Weight decay for AdamW optimizer.")
    return parser.parse_args()

def psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return -10.0 * math.log10(max(mse, 1.0e-12))

# --- Charbonnier Loss (Smooth L1) ---
def calculate_custom_loss(pred, target, args):
    eps = 1e-3
    
    r_pred = pred[:, 0:1]; g_pred = pred[:, 1:2]; b_pred = pred[:, 2:3]
    r_gt   = target[:, 0:1]; g_gt   = target[:, 1:2]; b_gt   = target[:, 2:3]

    l_g  = torch.mean(torch.sqrt((g_pred - g_gt)**2 + eps**2))
    l_rg = torch.mean(torch.sqrt(((r_pred - g_pred) - (r_gt - g_gt))**2 + eps**2))
    l_bg = torch.mean(torch.sqrt(((b_pred - g_pred) - (b_gt - g_gt))**2 + eps**2))
    
    total_loss = (args.weight_g * l_g + args.weight_rg * l_rg + args.weight_bg * l_bg)
    
    return total_loss, l_rg, l_g, l_bg

def run_epoch(model, loader, optimizer, scaler, args, device, train, epoch, current_lr=None):
    model.train(train)
    total_loss = 0.0
    total_psnr = 0.0
    sum_chroma_rg, sum_luma_g, sum_chroma_bg = 0.0, 0.0, 0.0
    
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        autocast = torch.amp.autocast
    else:
        autocast = torch.cuda.amp.autocast

    visual_saved_this_epoch = False
    lr_str = f" | LR: {current_lr:.6f}" if current_lr is not None else ""

    for x, y, path, noise_level in loader:
        x = x.to(device)
        y = y.to(device)
        
        noise_level = noise_level.to(device).float()
        if noise_level.dim() == 0:
            noise_level = noise_level.view(1, 1, 1, 1).expand(x.shape[0], -1, -1, -1)
        else:
            noise_level = noise_level.view(-1, 1, 1, 1)
        
        with torch.set_grad_enabled(train):
            with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                pred = model(x, noise_level)
                loss, l_rg, l_g, l_bg = calculate_custom_loss(pred, y, args)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                
                if args.amp and device.type == "cuda":
                    try:
                        scaler.unscale_(optimizer)
                    except AttributeError:
                        pass 
                    
                # Increased threshold to 5.0 for stable deep-network training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                
                scaler.step(optimizer)
                scaler.update()
        total_loss += loss.item()
        
        # PSNR handles clamping internally via MSE, suitable for linear outputs
        total_psnr += psnr(pred.detach(), y) 
        
        sum_chroma_rg += l_rg.item()
        sum_luma_g    += l_g.item()
        sum_chroma_bg += l_bg.item()

        if not train and args.save_val_visual and args.val_scene_name:
            path_str = str(path).lower().replace("\\", "/")
            scene_query = args.val_scene_name.lower()
            
            if scene_query in path_str:
                visual_saved_this_epoch = True
                
                if args.save_every == 0 or epoch % args.save_every == 0:
                    pred_np = pred.detach().cpu().numpy()
                    gt_np   = y.detach().cpu().numpy()

                    if pred_np.shape[1] == 3 and pred_np.shape[2] == args.patch_size:
                        p_data = np.transpose(pred_np, (0, 2, 3, 1))
                        g_data = np.transpose(gt_np,   (0, 2, 3, 1))

                        batch_size = p_data.shape[0]
                        grid_h = int(math.sqrt(batch_size))
                        if grid_h == 0: grid_h = 1
                        grid_w = batch_size // grid_h
                        if grid_w == 0: grid_w = 1
                        
                        num_to_use = grid_h * grid_w
                        if num_to_use > batch_size:
                            grid_h, grid_w = 1, batch_size
                            num_to_use = batch_size
                        
                        p_data = p_data[:num_to_use]
                        g_data = g_data[:num_to_use]

                        grid_pred = p_data.reshape(grid_h, grid_w, args.patch_size, args.patch_size, 3)
                        grid_gt   = g_data.reshape(grid_h, grid_w, args.patch_size, args.patch_size, 3)

                        grid_pred = grid_pred.transpose(0, 2, 1, 3, 4)
                        grid_gt   = grid_gt.transpose(0, 2, 1, 3, 4)

                        final_pred = grid_pred.reshape(grid_h * args.patch_size, grid_w * args.patch_size, 3)
                        final_gt   = grid_gt.reshape(grid_h * args.patch_size, grid_w * args.patch_size, 3)

                        out_dir = Path(args.out_dir).resolve()
                        
                        save_path_gt = out_dir / f"gt_val_{args.val_scene_name}.png"
                        if not save_path_gt.exists(): 
                            img_gt = Image.fromarray((np.clip(final_gt, 0, 1) * 255).astype(np.uint8))
                            img_gt.save(save_path_gt)

                        iso_suffix = f"_iso{args.val_iso}" if hasattr(args, 'val_iso') and args.val_iso else ""
                        save_path_pred = out_dir / f"pred{iso_suffix}_epoch_{epoch:03d}.png"
                        
                        img_pred = Image.fromarray((np.clip(final_pred, 0, 1) * 255).astype(np.uint8))
                        img_pred.save(save_path_pred)

    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)
    
    prefix = "Train" if train else f"Val (ISO {args.val_iso})"
    print(f"{prefix} | Epoch {epoch:03d} | PSNR: {avg_psnr:.2f} | Loss: {avg_loss:.6f} | Luma(G): {sum_luma_g/len(loader):.6f} | Chroma(RG): {sum_chroma_rg/len(loader):.6f} | Chroma(BG): {sum_chroma_bg/len(loader):.6f}{lr_str}")
    
    if not train and args.save_val_visual and not visual_saved_this_epoch:
        print(f"[VISUALS] WARNING: Could not find patch '{args.val_scene_name}' in validation set.")

    return avg_loss, avg_psnr


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    args.out_dir = Path(args.out_dir).resolve()
    print(f"Output Directory: {args.out_dir}")

    data_root = Path(args.data_root)
    mode = args.dataset_mode
    if mode == "auto":
        mode = "folders" if (data_root / "train").exists() and (data_root / "val").exists() else "unified_patches"

    hard_patches_path = resolve_hard_patches(args)
    
    train_paths = list_unified_patches(
        data_root, "train", hard_patches_path, 
        max_images=args.max_train_images, gt_only=True 
    )
    val_paths = list_unified_patches(
        data_root, "val", None, 
        max_images=args.max_val_images, gt_only=True
    )

    if not train_paths or not val_paths:
        raise ValueError("Expected non-empty train/val samples under {}".format(args.data_root))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.val_iso = args.val_iso_levels[0] 

    train_loader = DataLoader(
        QuadBayerDataset(train_paths, args, train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        QuadBayerDataset(val_paths, args, train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    model = build_model(args, debug_logging=args.debug_film).to(device)
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params}, 
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=args.lr, weight_decay=args.weight_decay)
    
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    best_psnr = -1.0

    start_epoch = 1
    checkpoint_path = out_dir / "best.pt"

    print("Device:", device)
    print("Dataset mode:", mode)
    if mode == "unified_patches":
        print("Hard patches:", str(resolve_hard_patches(args)) if resolve_hard_patches(args) else "none")
    print("Input channels:", make_input_channels(args))
    print("Model: simplified_naf (Linear Output)")
    print("Width:", args.width, "Blocks:", args.blocks, "Downsample:", args.downsample)
    print("Train images:", len(train_paths), "Val images:", len(val_paths))
    print("Params:", sum(p.numel() for p in model.parameters()))
    print("Max ISO Training Level:", args.max_iso)
    print("Validation ISO Levels:", args.val_iso_levels)

    if args.resume and checkpoint_path.exists():
        print(f"Attempting to resume from {checkpoint_path}...")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        
        for pg in optimizer.param_groups:
            pg['lr'] = args.lr
            
        scaler.load_state_dict(state["scaler"])
        best_psnr = state["val_psnr"]
        start_epoch = state["epoch"] + 1
        print(f"Loaded epoch: {state['epoch']}, Best PSNR: {best_psnr}")
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(args.epochs - start_epoch + 1), eta_min=args.lr_end
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_end
        )


    for epoch in range(start_epoch, args.epochs + 1):
        
        train_loader.dataset.set_epoch(epoch)

        train_loss, train_psnr = run_epoch(model, train_loader, optimizer, scaler, args, device, train=True, epoch=epoch)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        args.val_iso = args.val_iso_levels[0] 
        
        old_save_vis = args.save_val_visual
        if (args.save_every > 0 and epoch % args.save_every == 0):
            args.save_val_visual = False 

        run_epoch(model, val_loader, optimizer, scaler, args, device, train=False, epoch=epoch, current_lr=current_lr)

        args.save_val_visual = old_save_vis


        if args.save_every > 0 and (epoch % args.save_every == 0):
            print(f"\n--- Epoch {epoch}: Multi-Level Validation (LR: {current_lr:.6f}) ---")
            val_results_psnr = []
            
            for iso_idx, iso_level in enumerate(args.val_iso_levels):
                args.val_iso = iso_level

                val_loss, val_psnr = run_epoch(model, val_loader, optimizer, scaler, args, device, train=False, epoch=epoch)
                
                print(f"  ISO {iso_level:5d} | PSNR: {val_psnr:.2f}")
                val_results_psnr.append(val_psnr)
            
            avg_val_psnr = np.mean(val_results_psnr)
            
            print(f"  Average Val PSNR: {avg_val_psnr:.2f}")
            
            if avg_val_psnr > best_psnr:
                best_psnr = avg_val_psnr
                print(f"  New Best Avg PSNR -> Saving Checkpoint")
                
                state = {
                    "model": model.state_dict(), 
                    "args": vars(args), 
                    "epoch": epoch, 
                    "val_psnr": best_psnr, 
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict()
                }
                torch.save(state, checkpoint_path)

        elif args.save_every == 0:
             val_loss, val_psnr = run_epoch(model, val_loader, optimizer, scaler, args, device, train=False, epoch=epoch)

if __name__ == "__main__":
    main()
