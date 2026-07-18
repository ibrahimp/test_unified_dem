import numpy as np
from PIL import Image
from pathlib import Path
import random
import pickle
import torch
import math
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PATTERNS = {"grbg", "rggb", "gbrg", "bggr"}
COLOR_TO_CHANNEL = {"r": 0, "g": 1, "b": 2}

UNIFIED_SPLITS = {
    "train": [3, 4, 5, 6, 9, 10, 11, 12, 13, 14],
    "val": [1, 2], 
    "test": [7, 8, 15, 16, 17],
}

def list_images(folder):
    folder = Path(folder)
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTS)

def unified_patch_key(path, data_root):
    rel = Path(path).relative_to(data_root)
    scene_view = rel.parts[:2] 
    name_parts = rel.name.split("_")[:5]
    return "/".join((*scene_view, "_".join(name_parts))).replace("\\", "/")

def load_hard_patch_keys(path):
    with open(path, "rb") as handle:
        keys = pickle.load(handle)
        return set(str(k).replace("\\", "/") for k in keys)

def resolve_hard_patches(args):
    if args.hard_patches:
        return Path(args.hard_patches)
    if args.hard_patch_percentile is not None:
        return Path(args.data_root) / "hardpatches{:.2f}.pkl".format(args.hard_patch_percentile)
    return None

def list_unified_patches(data_root, split, hard_patches_path=None, max_images=0, gt_only=False):
    data_root = Path(data_root)
    hard_keys = load_hard_patch_keys(hard_patches_path) if hard_patches_path else None
    
    targets = []
    for s in UNIFIED_SPLITS[split]:
        p1 = data_root / f"Scene{s}"
        if p1.exists() and p1.is_dir(): 
            targets.append(p1)

    if not targets:
        print(f"WARNING: No specific {split} scenes found. Scanning for any available scenes.")
        for p in data_root.glob("Scene*"):
            if p.is_dir(): targets.append(p)

    patch_groups = {}
    for scene_dir in targets:
        all_files = list(scene_dir.rglob("*_gt.npy")) + list(scene_dir.rglob("*_iso*.npy"))
        
        for f in all_files:
            name = f.name
            patch_id = None
            
            if "_gt.npy" in name:
                patch_id = name.split("_gt.npy")[0]
            elif "_iso" in name:
                parts = name.split("_iso")
                patch_id = parts[0]
            
            if patch_id:
                rel_parent = f.parent.relative_to(data_root).as_posix() 
                group_key = f"{rel_parent}/{patch_id}"
                
                if group_key not in patch_groups:
                    patch_groups[group_key] = {"gt": None, "noisy": []}
                
                if "_gt.npy" in name:
                    patch_groups[group_key]["gt"] = f
                else:
                    patch_groups[group_key]["noisy"].append(f)

    pairs = []
    
    for key, data in patch_groups.items():
        if hard_keys is not None:
            gt_path_str = str(data["gt"].relative_to(data_root)).replace("\\", "/") if data["gt"] else ""
            is_hard = (gt_path_str in hard_keys)

            if not is_hard:
                continue 

        if data["gt"]:
            pairs.append((data["gt"], data["gt"])) 
            
    print(f"DEBUG [{split}]: Found {len(pairs)} valid patches.")
    
    return pairs[:max_images] if max_images else pairs


# --- SYMMETRIC NOISE MODEL (No Clipping) ---
def apply_poisson_gaussian_noise(raw_signal, iso_level, npy_scale=13496.0):
    full_well_capacity = float(npy_scale) 
    signal_dn = raw_signal * full_well_capacity
    
    gain = iso_level / 100.0
    shot_variance = signal_dn 
    base_read_noise_std = 5.0 
    read_var = (base_read_noise_std * gain)**2
    
    total_variance = shot_variance + read_var
    total_std = np.sqrt(total_variance)
    
    noise = np.random.normal(0, total_std, size=signal_dn.shape).astype(np.float32)
    noisy_dn = signal_dn + noise
    
    # No clipping: allows symmetric negative excursions for true zero recovery
    return noisy_dn / full_well_capacity

# --- END NOISE MODEL ---

def load_data(path, npy_scale=13496.0):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        data = np.load(path).astype(np.float32)
        
        if data.ndim == 3:
            if data.shape[0] == 3 and data.shape[1] != 3:
                data = np.transpose(data, (1, 2, 0))
            return np.clip(data / npy_scale, 0.0, 1.0)
        
        if data.ndim == 2:
            return data / npy_scale
        
        return data / npy_scale
    
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0

# --- HELPER FUNCTIONS ---

def crop_at(data, y, x, patch_size):
    if data.ndim == 2:
        return data[y : y + patch_size, x : x + patch_size]
    else:
        return data[y : y + patch_size, x : x + patch_size, :]

def paired_crop(rgb_1, rgb_2, patch_size, randomize):
    height = min(rgb_1.shape[0], rgb_2.shape[0])
    width = min(rgb_1.shape[1], rgb_2.shape[1])

    y_max = max(0, height - patch_size)
    x_max = max(0, width - patch_size)

    if randomize:
        y = random.randint(0, y_max)
        x = random.randint(0, x_max)
    else:
        y = y_max // 2
        x = x_max // 2
        
    return crop_at(rgb_1, y, x, patch_size), crop_at(rgb_2, y, x, patch_size)

def align_size(size, factor):
    return max(factor, (size // factor) * factor)

def ensure_factor_aligned(data, factor):
    height, width = data.shape[:2]
    aligned_height = align_size(height, factor)
    aligned_width = align_size(width, factor)
    if data.ndim == 3:
        return data[:aligned_height, :aligned_width, :]
    else:
        return data[:aligned_height, :aligned_width]

def augment(data):
    if random.random() < 0.5: data = np.flip(data, axis=1)
    if random.random() < 0.5: data = np.flip(data, axis=0)
    k = random.randint(0, 3)
    if k: data = np.rot90(data, k, axes=(0, 1))
    return np.ascontiguousarray(data)

def paired_augment(rgb_1, rgb_2):
    if random.random() < 0.5:
        rgb_1 = np.flip(rgb_1, axis=1); rgb_2 = np.flip(rgb_2, axis=1)
    if random.random() < 0.5:
        rgb_1 = np.flip(rgb_1, axis=0); rgb_2 = np.flip(rgb_2, axis=0)
    k = random.randint(0, 3)
    if k:
        rgb_1 = np.rot90(rgb_1, k, axes=(0, 1)); rgb_2 = np.rot90(rgb_2, k, axes=(0, 1))
    return np.ascontiguousarray(rgb_1), np.ascontiguousarray(rgb_2)

def pattern_color(pattern, y, x, mode):
    if mode == "bayer":
        return pattern[(y % 2) * 2 + (x % 2)]
    else:
        return pattern[((y // 2) % 2) * 2 + ((x // 2) % 2)]

def make_cfa_masks(height, width, pattern, mode):
    masks = np.zeros((3, height, width), dtype=np.float32)
    if mode == "bayer":
        for y_mod in range(2):
            for x_mod in range(2):
                channel = COLOR_TO_CHANNEL[pattern[y_mod * 2 + x_mod]]
                masks[channel, y_mod::2, x_mod::2] = 1.0
    else:
        for y_block in range(0, 4, 2):
            for x_block in range(0, 4, 2):
                channel = COLOR_TO_CHANNEL[pattern_color(pattern, y_block, x_block, mode)]
                for y_offset in range(y_block, y_block + 2):
                    for x_offset in range(x_block, x_block + 2):
                        masks[channel, y_offset::4, x_offset::4] = 1.0
    return masks

def rgb_to_bayer(rgb, pattern, mode):
    height, width, _ = rgb.shape
    
    raw = np.zeros((height, width), dtype=np.float32) 
    
    if mode == "bayer":
        raw[0::2, 0::2] = rgb[0::2, 0::2, COLOR_TO_CHANNEL[pattern[0]]]
        raw[0::2, 1::2] = rgb[0::2, 1::2, COLOR_TO_CHANNEL[pattern[1]]]
        raw[1::2, 0::2] = rgb[1::2, 0::2, COLOR_TO_CHANNEL[pattern[2]]]
        raw[1::2, 1::2] = rgb[1::2, 1::2, COLOR_TO_CHANNEL[pattern[3]]]
    else: # Quad Bayer
        for y_block in range(0, height, 4):
            for x_block in range(0, width, 4):
                h_clip = min(2, height - y_block)
                w_clip = min(2, width - x_block)
                
                raw[y_block : y_block+h_clip, x_block : x_block+w_clip] = \
                    rgb[y_block : y_block+h_clip, x_block : x_block+w_clip, COLOR_TO_CHANNEL[pattern[0]]]
                raw[y_block : y_block+h_clip, x_block+2 : x_block+4] = \
                    rgb[y_block : y_block+h_clip, x_block+2 : x_block+4, COLOR_TO_CHANNEL[pattern[1]]]
                raw[y_block+2 : y_block+4, x_block : x_block+2] = \
                    rgb[y_block+2 : y_block+4, x_block : x_block+2, COLOR_TO_CHANNEL[pattern[2]]]
                raw[y_block+2 : y_block+4, x_block+2 : x_block+4] = \
                    rgb[y_block+2 : y_block+4, x_block+2 : x_block+4, COLOR_TO_CHANNEL[pattern[3]]]
    return raw

class QuadBayerDataset(Dataset):
    def __init__(self, paths, args, train):
        self.paths = paths
        self.args = args
        self.train = train
        self.active_patterns = list(PATTERNS) if args.multi_pattern else [args.pattern]
        
        # Pedestal removed: linear output + symmetric noise handles true zeros natively
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def _get_curriculum_max_iso(self):
        if not self.train: return self.args.max_iso
        
        if self.current_epoch <= 20:
            return min(400, self.args.max_iso)
        
        progress = (self.current_epoch - 20) / 40.0 
        ramped_iso = int(400 + (progress * (self.args.max_iso - 400)))
        return min(ramped_iso, self.args.max_iso)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        _, gt_path = self.paths[index] 
        rgb_gt = load_data(gt_path, self.args.npy_scale)
        
        patch_size = align_size(self.args.patch_size, self.args.downsample)

        if self.train:
            max_target_iso = self._get_curriculum_max_iso()
            target_iso = random.randint(100, max_target_iso)
            
            rgb_gt_cropped, _ = paired_crop(rgb_gt, rgb_gt, patch_size, randomize=True)
            rgb_gt_cropped, _ = paired_augment(rgb_gt_cropped, rgb_gt_cropped)
        else:
            target_iso = self.args.val_iso
            rgb_gt_cropped, _ = paired_crop(rgb_gt, rgb_gt, patch_size, randomize=False)

        pattern = self.args.pattern
        if self.train and self.args.multi_pattern:
            pattern = random.choice(self.active_patterns)
        
        generation_mode = "quad_bayer" if self.args.binning > 1 else self.args.mode
        
        raw_mosaic_clean = rgb_to_bayer(rgb_gt_cropped, pattern, generation_mode)
        
        # Apply symmetric noise BEFORE binning
        raw_mosaic_noisy = apply_poisson_gaussian_noise(raw_mosaic_clean, target_iso, self.args.npy_scale)

        if self.args.binning > 1:
            h, w = raw_mosaic_noisy.shape
            raw_binned = raw_mosaic_noisy.reshape(h//self.args.binning, self.args.binning, 
                                                  w//self.args.binning, self.args.binning).mean(axis=(1, 3)).astype(np.float32)
        else:
            raw_binned = raw_mosaic_noisy

        raw_input_noisy = raw_binned
        
        iso_norm = (math.log2(target_iso) - math.log2(100)) / (math.log2(self.args.max_iso) - math.log2(100))
        
        raw_input_noisy = ensure_factor_aligned(raw_input_noisy, self.args.downsample)
        rgb_gt_cropped = ensure_factor_aligned(rgb_gt_cropped, self.args.downsample)

        height, width = raw_input_noisy.shape
        
        masks = make_cfa_masks(height, width, pattern, "bayer") 
        
        inputs = [raw_input_noisy[None]]
        if self.args.use_cfa:
            inputs.append(masks)
            
        # No pedestal injection. Raw symmetric values flow directly to the linear network.
        x = np.concatenate(inputs, axis=0).astype(np.float32)
        y = np.transpose(rgb_gt_cropped, (2, 0, 1)).astype(np.float32)
        
        return torch.from_numpy(x), torch.from_numpy(y), str(gt_path), torch.tensor(iso_norm)
