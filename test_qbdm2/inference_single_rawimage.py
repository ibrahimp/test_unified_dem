#!/usr/bin/env python3
"""
Inference Script for Single Raw Image (.raw format).
- Matches Final Model State: Linear Output, No Pedestal, Symmetric Noise.
- Normalizes input by 'npy_scale / 4' to match training distribution assumptions.
- Visualizes strictly as standard RAW data (Data As Is).
"""

import argparse
import math
from pathlib import Path
import numpy as np
import torch
import os
from PIL import Image

# --- IMPORTS FROM EXISTING FILES ---
try:
    from dataset import make_cfa_masks, COLOR_TO_CHANNEL
    from model import build_model
except ImportError as e:
    print(f"Error importing project files. Ensure 'dataset.py' and 'model.py' are in the same directory.\n{e}")
    exit(1)

def apply_gamma(img_array):
    """Applies sRGB gamma correction (approx)."""
    img_clipped = np.clip(np.asarray(img_array), 0.0, 1.0)
    return (img_clipped ** (1/2.2) * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="Single Image Inference (.raw)")
    
    # Input Arguments
    parser.add_argument("--input", type=str, required=True, help="Path to input .raw file")
    parser.add_argument("--width", type=int, required=True, help="Width of the raw image")
    parser.add_argument("--height", type=int, required=True, help="Height of the raw image")
    
    # --- NEW FLAGS ---
    parser.add_argument("--input-bits", type=int, default=12, help="Input raw bit depth (default: 12)")
    parser.add_argument("--output-filename", type=str, default=None, help="Custom output filename stem (without extension). Defaults to input filename.")
    
    # Model Arguments
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.")
    parser.add_argument("--iso", type=float, default=800.0, help="ISO level for noise conditioning (default 800).")
    
    # Output Arguments
    parser.add_argument("--output-dir", type=str, default="./inference_output")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- LOAD CHECKPOINT & ARGS ---
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if not ckpt_path or not ckpt_path.exists():
        runs_dir = Path("runs")
        for pt_file in runs_dir.rglob("best.pt"):
            ckpt_path = pt_file; break
            
    if not ckpt_path: raise FileNotFoundError("Could not find 'best.pt'.")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    saved_args = argparse.Namespace(**state["args"]) if "args" in state else None
    
    # --- SCALE ASSUMPTION (npy_scale / 4) ---
    npy_scale = getattr(saved_args, 'npy_scale', 13496.0) if saved_args else 13496.0
    effective_scale = npy_scale / 4.0 
    
    pattern_name = getattr(saved_args, 'pattern', 'grbg')

    model = build_model(saved_args).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    
    print(f"Model Loaded: Pattern={pattern_name}")
    print(f"Inference Config: Normalizing by effective scale ({effective_scale:.2f}), Bit Depth: {args.input_bits}-bit")

    # --- LOAD RAW FILE ---
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading raw data from {input_path} ({args.width}x{args.height})...")
    
    with open(input_path, 'rb') as f:
        # Reads as uint16 (standard for unpacked RAW in this pipeline)
        raw_data = np.fromfile(f, dtype=np.uint16).reshape(args.height, args.width)

    print(f"[Input Stats] Raw Data Min: {raw_data.min()}, Max: {raw_data.max()}")
    
    # --- NORMALIZATION (Using effective_scale) ---
    rgb_noisy = raw_data.astype(np.float32) / effective_scale
    
    H_full, W_full = rgb_noisy.shape
    
    # Ensure input is even dimensions for clean binning/downsampling
    if H_full % 2 != 0: 
        print(f"Warning: Height is odd. Cropping to {H_full-1}.")
        rgb_noisy = rgb_noisy[:H_full-1, :]
        H_full -= 1
    if W_full % 2 != 0: 
        print(f"Warning: Width is odd. Cropping to {W_full-1}.")
        rgb_noisy = rgb_noisy[:, :W_full-1]
        W_full -= 1

    # Step 1: Binning (Average 2x2 blocks) if configured in checkpoint 
    binning_factor = getattr(saved_args, 'binning', 1)
    
    if binning_factor > 1:
        h_bin, w_bin = H_full // binning_factor, W_full // binning_factor
        raw_binned = rgb_noisy.reshape(
            h_bin, binning_factor, 
            w_bin, binning_factor
        ).mean(axis=(1,3)).astype(np.float32)
    else:
        raw_binned = rgb_noisy

    print(f"Processing Shape: Input ({H_full}x{W_full}) -> Binned/Processed ({raw_binned.shape[0]}x{raw_binned.shape[1]})")

    # --- TILING CONFIGURATION (Robustness for any size) ---
    tile_size_in = 1024  
    overlap_in = 64      
    stride_in = tile_size_in - overlap_in 

    # Pad binned input so dimensions are multiples of tile_size
    pad_h = (tile_size_in - raw_binned.shape[0] % tile_size_in) % tile_size_in
    pad_w = (tile_size_in - raw_binned.shape[1] % tile_size_in) % tile_size_in
    
    raw_padded = np.pad(raw_binned, ((0, pad_h), (0, pad_w)), mode='edge')
    
    H_pad_total = raw_binned.shape[0] + pad_h
    W_pad_total = raw_binned.shape[1] + pad_w

    # Prepare Masks for the padded binned grid 
    mask_mode = "bayer" 
    cfa_masks_padded = make_cfa_masks(H_pad_total, W_pad_total, pattern_name.lower(), mask_mode)

    # Output buffers at Full Resolution (Accounting for Binning Upscale factor if applicable)
    out_H_full_padded = int(H_pad_total * binning_factor)
    out_W_full_padded = int(W_pad_total * binning_factor)
    
    output_buffer = np.zeros((3, out_H_full_padded, out_W_full_padded), dtype=np.float32)
    weight_map = np.zeros((out_H_full_padded, out_W_full_padded), dtype=np.float32)

    # --- INFERENCE LOOP ---
    print("Running Inference...")
    
    for y in range(0, H_pad_total - tile_size_in + 1, stride_in):
        for x in range(0, W_pad_total - tile_size_in + 1, stride_in):
            
            # Extract Tile (Bayer) and Mask
            tile_raw = raw_padded[y:y+tile_size_in, x:x+tile_size_in]
            tile_mask = cfa_masks_padded[:, y:y+tile_size_in, x:x+tile_size_in]

            # Prepare inputs [Raw, Masks...] -> Shape (1+Cm, H, W)
            raw_expanded = np.expand_dims(tile_raw, 0)
            combined_input = np.concatenate([raw_expanded, tile_mask], axis=0)

            # Noise Scalar Calculation
            max_iso = getattr(saved_args, 'max_iso', 12800) if saved_args else 12800
            
            safe_denom = math.log2(max_iso) - math.log2(100)
            iso_norm_val = (math.log2(args.iso) - math.log2(100)) / max(safe_denom, 1e-6)

            with torch.no_grad():
                x_tensor = torch.from_numpy(combined_input[np.newaxis]).to(device).float() 
                noise_tensor = torch.tensor(iso_norm_val).view(-1, 1, 1, 1).to(device)
                
                pred_tile = model(x_tensor, noise_tensor)

            # Move to CPU (Shape: 3, H_out, W_out where H_out is ~binning_factor * tile_size_in)
            pred_np = pred_tile[0].cpu().numpy()

            # Map coordinates back to Full Res Buffer 
            y_out = int(y * binning_factor) 
            x_out = int(x * binning_factor)
            
            h_p, w_p = pred_np.shape[1], pred_np.shape[2]
            
            output_buffer[:, y_out:y_out+h_p, x_out:x_out+w_p] += pred_np
            weight_map[y_out:y_out+h_p, x_out:x_out+w_p] += 1.0

    # --- POST-PROCESSING ---
    
    # Average overlaps
    final_pred = output_buffer / np.maximum(weight_map[None], 1e-8)
    
    # Crop back to original dimensions (remove padding and match input size)
    final_pred_crop = final_pred[:, :H_full, :W_full]

    # Transpose to HWC for saving
    final_rgb_hwc = np.transpose(final_pred_crop, (1, 2, 0))

    print(f"[Model Output Stats] Float Min: {final_rgb_hwc.min():.5f}, Max: {final_rgb_hwc.max():.5f}")

    # --- SAVE OUTPUT (.raw) ---
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Use custom filename if provided, otherwise fall back to input stem
    base_name = args.output_filename if args.output_filename else input_path.stem
    
    output_file_path = out_dir / f"{base_name}.raw"
    
    with open(output_file_path, 'wb') as f:
        # Dynamic max value based on bit depth flag
        max_raw_val = 2**args.input_bits - 1
        
        raw_out_uint16 = np.clip(final_rgb_hwc * effective_scale, 0.0, float(max_raw_val)).astype(np.uint16)
        
        print(f"[Final Output Stats] Min: {raw_out_uint16.min()}, Max: {raw_out_uint16.max()}")
        
        # Write interleaved RGB data (R,G,B,R,G,B...) as uint16
        raw_out_uint16.tofile(f)

    print(f"[Complete] Saved raw output to: {output_file_path}")

    # --- VISUALIZATION (PNG) - Data As Is ---
    
    # Normalize by the actual max value of the specified bit depth for correct display mapping
    vis_float = raw_out_uint16.astype(np.float32) / float(max_raw_val)
    
    img_out = apply_gamma(vis_float)
    png_path = out_dir / f"{base_name}_rgb_preview.png"
    
    Image.fromarray(img_out).save(png_path, "PNG")
    print(f"[Complete] Saved visualization to: {png_path}")

if __name__ == "__main__":
    main()
