"""
Run the unified demosaicing model on one image.

The pretrained model expects a 12-channel tensor made from single-, quad-,
and nona-Bayer mosaics plus RGB color masks. This script accepts either a
single-channel raw mosaic or an RGB image, builds that model input, and saves
the selected model head output.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from arch.simple_multi_head_jd_model import SIMP_MULTI_JD  # noqa: E402


DEFAULT_SENSOR_MAX = 13496.0


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image inference for unified demosaicing.")
    parser.add_argument("--input", required=True, help="Input image path (.npy, .png, .jpg, .tif, ...).")
    parser.add_argument(
        "--input-mode",
        choices=["mosaic", "quad-mosaic", "bayer-mosaic", "rgb"],
        default="quad-mosaic",
        help="Use mosaic for single-channel raw input, or rgb to synthesize mosaics from RGB. quad-mosaic and bayer-mosaic are aliases.",
    )
    parser.add_argument(
        "--pattern-type",
        choices=["bayer", "quad"],
        default="quad",
        help="Raw mosaic pattern type when --input-mode mosaic is used.",
    )
    parser.add_argument(
        "--mosaic-order",
        default="GRBG",
        help="2x2 Bayer order: RGGB, GRBG, GBRG, or BGGR. For quad pattern type, each Bayer position is a 2x2 block.",
    )
    parser.add_argument("--raw-width", type=int, default=None, help="Width for headerless .raw input files.")
    parser.add_argument("--raw-height", type=int, default=None, help="Height for headerless .raw input files.")
    parser.add_argument(
        "--raw-dtype",
        choices=["uint16", "uint8", "float32"],
        default="uint16",
        help="Element type for headerless .raw input files. Use uint16 for unpacked 10/12/14/16-bit raw.",
    )
    parser.add_argument(
        "--raw-byte-order",
        choices=["little", "big", "native"],
        default="little",
        help="Byte order for multi-byte headerless .raw input files.",
    )
    parser.add_argument(
        "--checkpoint",
        default="PyTorch/models/iso3200.pth",
        help="Checkpoint path. Shipped options include PyTorch/models/iso400.pth, iso800.pth, iso1600.pth, iso3200.pth.",
    )
    parser.add_argument("--output", default="outputs/single_image", help="Output file prefix or directory.")
    parser.add_argument(
        "--head",
        choices=["auto", "single", "quad", "nona", "all"],
        default="auto",
        help="Which model output head(s) to save. auto uses single for Bayer, quad for Quad Bayer, and all for RGB.",
    )
    parser.add_argument(
        "--sensor-max",
        type=float,
        default=DEFAULT_SENSOR_MAX,
        help="Raw-domain white level used by this repo for normalization.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs="+",
        default=[1024],
        help="Inference tile size. Use one value for square tiles or two values for height width. Use 0 to disable tiling.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=64,
        help="Overlap in pixels between neighboring tiles. Ignored when tiling is disabled.",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save an 8-bit sRGB-like PNG preview for each output.",
    )
    parser.add_argument(
        "--png-gamma",
        type=float,
        default=2.2,
        help="Gamma used for PNG previews. Use 1.0 for a linear preview.",
    )
    return parser.parse_args()


def scale_to_sensor_range(image, sensor_max):
    image = image.astype(np.float32)
    if image.max() <= 1.0:
        image *= sensor_max
    elif image.max() <= 255.0:
        image = image / 255.0 * sensor_max
    elif image.max() > sensor_max:
        dtype_max = 65535.0 if image.max() > 255.0 else 255.0
        image = image / dtype_max * sensor_max
    return np.clip(image, 0.0, sensor_max)


def raw_numpy_dtype(raw_dtype, byte_order):
    dtype = np.dtype(raw_dtype)
    if dtype.itemsize == 1 or byte_order == "native":
        return dtype
    return dtype.newbyteorder("<" if byte_order == "little" else ">")


def read_headerless_raw(path, raw_width, raw_height, raw_dtype, raw_byte_order):
    if raw_width is None or raw_height is None:
        raise ValueError("Headerless .raw input requires both --raw-width and --raw-height.")

    dtype = raw_numpy_dtype(raw_dtype, raw_byte_order)
    expected_items = raw_width * raw_height
    image = np.fromfile(path, dtype=dtype)
    if image.size != expected_items:
        expected_bytes = expected_items * dtype.itemsize
        actual_bytes = Path(path).stat().st_size
        raise ValueError(
            f"Raw file size does not match dimensions: got {actual_bytes} bytes / {image.size} items, "
            f"expected {expected_bytes} bytes / {expected_items} items for "
            f"{raw_width}x{raw_height} {raw_dtype}."
        )
    return image.reshape(raw_height, raw_width)


def read_image(path, raw_width=None, raw_height=None, raw_dtype="uint16", raw_byte_order="little"):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".raw":
        return read_headerless_raw(path, raw_width, raw_height, raw_dtype, raw_byte_order)

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read input image: {path}")
    return image


def read_rgb_image(path, sensor_max, raw_width=None, raw_height=None, raw_dtype="uint16", raw_byte_order="little"):
    image = read_image(path, raw_width, raw_height, raw_dtype, raw_byte_order)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected an HxWx3 RGB input, got shape {image.shape}.")
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif Path(path).suffix.lower() != ".npy":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return scale_to_sensor_range(image, sensor_max)


def read_mosaic_image(path, sensor_max, raw_width=None, raw_height=None, raw_dtype="uint16", raw_byte_order="little"):
    image = read_image(path, raw_width, raw_height, raw_dtype, raw_byte_order)
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            raise ValueError(f"Expected a single-channel mosaicked input, got shape {image.shape}.")
    return scale_to_sensor_range(image, sensor_max)


def pad_to_multiple(image, multiple):
    height, width = image.shape[:2]
    padded_height = int(np.ceil(height / multiple) * multiple)
    padded_width = int(np.ceil(width / multiple) * multiple)
    pad_bottom = padded_height - height
    pad_right = padded_width - width
    if pad_bottom == 0 and pad_right == 0:
        return image, (height, width)
    if image.ndim == 2:
        padded = np.pad(image, ((0, pad_bottom), (0, pad_right)), mode="reflect")
    else:
        padded = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="reflect")
    return padded, (height, width)


def build_pattern(height, width, filter_size, order="RGGB"):
    channel_for = {"R": 0, "G": 1, "B": 2}
    tile = np.empty((filter_size * 2, filter_size * 2), dtype=np.int64)
    for row in range(2):
        for col in range(2):
            channel = channel_for[order[row * 2 + col]]
            tile[
                row * filter_size : (row + 1) * filter_size,
                col * filter_size : (col + 1) * filter_size,
            ] = channel
    reps = (height // tile.shape[0], width // tile.shape[1])
    return np.tile(tile, reps)


def normalize_mosaic_order(order):
    order = order.upper().replace("-", "").replace("_", "")
    valid_orders = {"RGGB", "GRBG", "GBRG", "BGGR"}
    if order not in valid_orders:
        raise ValueError(f"Unsupported mosaic order '{order}'. Use one of: {', '.join(sorted(valid_orders))}.")
    return order


def construct_mosaic_from_rgb(rgb, filter_size, sensor_max, order="RGGB"):
    height, width, _ = rgb.shape
    pattern = build_pattern(height, width, filter_size, order=order)
    mosaic = np.take_along_axis(rgb, pattern[:, :, None], axis=2)[:, :, 0]
    return construct_mosaic_channels(mosaic, pattern, sensor_max)


def construct_mosaic_channels(mosaic, pattern, sensor_max):
    masks = [(pattern == channel).astype(np.float32) * sensor_max for channel in range(3)]
    return np.dstack([mosaic, *masks]).astype(np.float32)


def normalize_model_input(model_input, sensor_max):
    model_input = model_input.transpose(2, 0, 1)
    model_input = model_input / sensor_max
    model_input = (model_input - 0.5) / 0.5
    return torch.from_numpy(model_input).unsqueeze(0).float()


def preprocess_rgb_input(rgb, sensor_max):
    rgb, original_shape = pad_to_multiple(rgb, 6)
    mosaics = [construct_mosaic_from_rgb(rgb, filter_size, sensor_max) for filter_size in (1, 2, 3)]
    model_input = np.dstack(mosaics)
    return normalize_model_input(model_input, sensor_max), original_shape


def resolve_pattern_type(input_mode, pattern_type):
    if input_mode == "bayer-mosaic":
        return "bayer"
    if input_mode == "quad-mosaic":
        return "quad"
    return pattern_type


def preprocess_mosaic_input(mosaic, sensor_max, mosaic_order, pattern_type):
    filter_size = 1 if pattern_type == "bayer" else 2
    pad_multiple = filter_size * 2
    mosaic, original_shape = pad_to_multiple(mosaic, pad_multiple)
    height, width = mosaic.shape
    pattern = build_pattern(height, width, filter_size=filter_size, order=mosaic_order)
    mosaic_channels = construct_mosaic_channels(mosaic, pattern, sensor_max)

    empty_channels = np.zeros_like(mosaic_channels)
    if pattern_type == "bayer":
        model_input = np.dstack([mosaic_channels, empty_channels, empty_channels])
    else:
        model_input = np.dstack([empty_channels, mosaic_channels, empty_channels])
    return normalize_model_input(model_input, sensor_max), original_shape


def load_model(checkpoint_path, device):
    model = SIMP_MULTI_JD().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("net_state_dict", checkpoint)
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def tensor_to_uint16(output, original_shape, sensor_max):
    if output.ndim == 4:
        output = output.squeeze(0)
    output = output.detach().cpu().numpy()
    output = np.clip((output * 0.5 + 0.5) * sensor_max, 0, sensor_max)
    output = output.transpose(1, 2, 0)
    height, width = original_shape
    return output[:height, :width, :].round().astype(np.uint16)


def parse_tile_size(tile_size_arg):
    if len(tile_size_arg) == 1:
        return tile_size_arg[0], tile_size_arg[0]
    if len(tile_size_arg) == 2:
        return tile_size_arg[0], tile_size_arg[1]
    raise ValueError("--tile-size expects either one value or two values: height width.")


def tile_starts(length, tile_size, overlap):
    if tile_size <= 0 or tile_size >= length:
        return [0]
    stride = tile_size - overlap
    starts = [0]
    while starts[-1] + tile_size < length:
        next_start = starts[-1] + stride
        if next_start + tile_size >= length:
            next_start = length - tile_size
        if next_start == starts[-1]:
            break
        starts.append(next_start)
    return starts


def selected_heads(head, input_mode, pattern_type):
    if head == "all":
        return ("single", "quad", "nona")
    if head != "auto":
        return (head,)
    if input_mode == "rgb":
        return ("single", "quad", "nona")
    return ("single",) if pattern_type == "bayer" else ("quad",)


def run_model_full(model, model_input, heads, device):
    outputs = model(model_input.to(device))
    named_outputs = dict(zip(("single", "quad", "nona"), outputs))
    return {head: named_outputs[head].squeeze(0).detach().cpu() for head in heads}


def run_model_tiled(model, model_input, heads, device, tile_height, tile_width, overlap):
    _, _, height, width = model_input.shape

    if tile_height <= 0 or tile_width <= 0:
        return run_model_full(model, model_input, heads, device)

    tile_height = min(tile_height, height)
    tile_width = min(tile_width, width)
    overlap = min(overlap, tile_height - 1, tile_width - 1)
    if overlap < 0:
        overlap = 0
    if overlap >= tile_height or overlap >= tile_width:
        raise ValueError("--tile-overlap must be smaller than both tile height and tile width.")

    y_starts = tile_starts(height, tile_height, overlap)
    x_starts = tile_starts(width, tile_width, overlap)
    stitched = {head: torch.empty((3, height, width), dtype=torch.float32) for head in heads}
    total_tiles = len(y_starts) * len(x_starts)
    tile_index = 0

    for y_index, y0 in enumerate(y_starts):
        y1 = y0 + tile_height
        write_y0 = 0 if y_index == 0 else (y_starts[y_index - 1] + tile_height + y0) // 2
        write_y1 = height if y_index == len(y_starts) - 1 else (y0 + y_starts[y_index + 1] + tile_height) // 2
        crop_y0 = write_y0 - y0
        crop_y1 = write_y1 - y0

        for x_index, x0 in enumerate(x_starts):
            tile_index += 1
            x1 = x0 + tile_width
            write_x0 = 0 if x_index == 0 else (x_starts[x_index - 1] + tile_width + x0) // 2
            write_x1 = width if x_index == len(x_starts) - 1 else (x0 + x_starts[x_index + 1] + tile_width) // 2
            crop_x0 = write_x0 - x0
            crop_x1 = write_x1 - x0

            print(
                f"tile {tile_index}/{total_tiles}: "
                f"input rows {y0}:{y1}, cols {x0}:{x1}; "
                f"write rows {write_y0}:{write_y1}, cols {write_x0}:{write_x1}"
            )
            tile_input = model_input[:, :, y0:y1, x0:x1].to(device)
            outputs = model(tile_input)
            named_outputs = dict(zip(("single", "quad", "nona"), outputs))

            for head in heads:
                tile_output = named_outputs[head].squeeze(0).detach().cpu()
                stitched[head][
                    :,
                    write_y0:write_y1,
                    write_x0:write_x1,
                ] = tile_output[:, crop_y0:crop_y1, crop_x0:crop_x1]

            del tile_input, outputs, named_outputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return stitched


def output_paths(output_arg, input_path, head):
    output = Path(output_arg)
    if output.suffix:
        stem_path = output.with_suffix("")
    else:
        stem_path = output / Path(input_path).stem
    suffix = f"_{head}" if head != "all" else ""
    return stem_path.parent / f"{stem_path.name}{suffix}.npy", stem_path.parent / f"{stem_path.name}{suffix}.png"


def save_png_preview(path, image_u16, sensor_max, gamma):
    preview = np.clip(image_u16.astype(np.float32) / sensor_max, 0.0, 1.0)
    if gamma != 1.0:
        preview = preview ** (1.0 / gamma)
    preview = (preview * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    mosaic_order = normalize_mosaic_order(args.mosaic_order)
    if args.input_mode == "rgb":
        rgb = read_rgb_image(
            args.input,
            args.sensor_max,
            args.raw_width,
            args.raw_height,
            args.raw_dtype,
            args.raw_byte_order,
        )
        model_input, original_shape = preprocess_rgb_input(rgb, args.sensor_max)
    else:
        pattern_type = resolve_pattern_type(args.input_mode, args.pattern_type)
        mosaic = read_mosaic_image(
            args.input,
            args.sensor_max,
            args.raw_width,
            args.raw_height,
            args.raw_dtype,
            args.raw_byte_order,
        )
        model_input, original_shape = preprocess_mosaic_input(
            mosaic,
            args.sensor_max,
            mosaic_order,
            pattern_type,
        )

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    tile_height, tile_width = parse_tile_size(args.tile_size)
    pattern_type = resolve_pattern_type(args.input_mode, args.pattern_type)
    heads = selected_heads(args.head, args.input_mode, pattern_type)

    with torch.no_grad():
        named_outputs = run_model_tiled(
            model,
            model_input,
            heads,
            device,
            tile_height,
            tile_width,
            args.tile_overlap,
        )

    for head in heads:
        image_u16 = tensor_to_uint16(named_outputs[head], original_shape, args.sensor_max)
        npy_path, png_path = output_paths(args.output, args.input, head)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, image_u16)
        if args.save_png:
            save_png_preview(png_path, image_u16, args.sensor_max, args.png_gamma)
        print(f"saved {head}: {npy_path}")
        if args.save_png:
            print(f"saved {head} preview: {png_path}")


if __name__ == "__main__":
    main()
