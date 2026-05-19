#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run one 10-bit uint16 RAW Bayer frame through a trained JDNDMSR model.

The network was trained on RGGB mosaics normalized to [-1, 1]. This script aligns
other Bayer orders before inference, then writes both a viewable RGB image and a
remosaiced uint16 RAW file in the requested output pattern.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

try:
    tf.compat.v1.disable_eager_execution()
except AttributeError:
    pass

from network.network import get_model


RGGB_OFFSETS = {
    "rggb": (0, 0),
    "grbg": (0, 1),
    "gbrg": (1, 0),
    "bggr": (1, 1),
}

COLOR_INDEX = {"b": 0, "g": 1, "r": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-image RAW inference for End-to-End-JDNDMSR."
    )
    parser.add_argument("--input", required=True, help="Input .raw file, uint16 little-endian by default.")
    parser.add_argument("--width", type=int, required=True, help="Input RAW width in pixels.")
    parser.add_argument("--height", type=int, required=True, help="Input RAW height in pixels.")
    parser.add_argument("--output-dir", default="raw_infer_results", help="Directory for outputs.")
    parser.add_argument("--output-name", default=None, help="Base name for output files.")
    parser.add_argument("--model", default="models/jdndmsr+_model.h5", help="Trained .h5 weights path.")
    parser.add_argument("--pattern", default="grbg", choices=sorted(RGGB_OFFSETS), help="Input/output Bayer order.")
    parser.add_argument("--output-pattern", default=None, choices=sorted(RGGB_OFFSETS), help="RAW output order. Defaults to --pattern.")
    parser.add_argument("--bit-depth", type=int, default=10, help="Valid signal bits in the uint16 RAW.")
    parser.add_argument("--byte-order", default="little", choices=["little", "big"], help="Input/output uint16 byte order.")
    parser.add_argument("--scale-factor", type=int, default=2, choices=[1, 2, 3, 4], help="Model super-resolution scale factor.")
    parser.add_argument("--noise", type=float, default=10.0, help="Estimated noise level in input DN at --bit-depth.")
    parser.add_argument("--layers", type=int, default=4, help="Number of residual groups used by the checkpoint.")
    parser.add_argument("--filters", type=int, default=64, help="Number of model filters used by the checkpoint.")
    parser.add_argument("--tile-size", type=int, default=0, help="Input tile size. Use 0 for whole-frame inference.")
    parser.add_argument("--tile-overlap", type=int, default=64, help="Input overlap for tiled inference.")
    parser.add_argument("--png", action="store_true", help="Kept for compatibility; an 8-bit PNG preview is always written.")
    return parser.parse_args()


def read_raw(path, width, height, byte_order):
    dtype = np.dtype("<u2" if byte_order == "little" else ">u2")
    raw = np.fromfile(path, dtype=dtype)
    expected = width * height
    if raw.size != expected:
        raise ValueError("Expected {} pixels, found {} in {}".format(expected, raw.size, path))
    return raw.reshape((height, width)).astype(np.uint16)


def write_raw(path, image, byte_order):
    dtype = np.dtype("<u2" if byte_order == "little" else ">u2")
    image.astype(dtype, copy=False).tofile(path)


def align_to_rggb(raw, pattern):
    y_offset, x_offset = RGGB_OFFSETS[pattern]
    return np.roll(raw, shift=(-y_offset, -x_offset), axis=(0, 1))


def undo_alignment(rgb, pattern, scale_factor):
    y_offset, x_offset = RGGB_OFFSETS[pattern]
    return np.roll(rgb, shift=(y_offset * scale_factor, x_offset * scale_factor), axis=(0, 1))


def make_model(args, add_noise):
    model = get_model(
        "adam",
        "he_normal",
        "mean_absolute_error",
        args.filters,
        args.layers,
        args.scale_factor,
        add_noise,
    )
    model.load_weights(args.model)
    return model


def predict_array(model, mosaic, noise_level, max_value, add_noise):
    height, width = mosaic.shape
    even_height = height - (height % 2)
    even_width = width - (width % 2)
    mosaic = mosaic[:even_height, :even_width]
    x = mosaic.astype(np.float32) / float(max_value)
    x = (x - 0.5) / 0.5
    x = x[np.newaxis, :, :, np.newaxis]
    if add_noise:
        noise = (noise_level / float(max_value)) * np.ones((1, even_height // 2, even_width // 2, 1), dtype=np.float32)
        pred = model.predict([x, noise], batch_size=1)
    else:
        pred = model.predict(x, batch_size=1)
    pred = np.clip(pred[0], -1.0, 1.0)
    return pred * 0.5 + 0.5


def blend_window(height, width):
    wy = np.hanning(height) if height > 2 else np.ones(height)
    wx = np.hanning(width) if width > 2 else np.ones(width)
    window = np.outer(np.maximum(wy, 0.05), np.maximum(wx, 0.05))
    return window[:, :, np.newaxis].astype(np.float32)


def predict_tiled(model, mosaic, args, max_value, add_noise):
    tile_size = args.tile_size
    overlap = args.tile_overlap
    if tile_size <= 0:
        return predict_array(model, mosaic, args.noise, max_value, add_noise)
    if tile_size <= overlap:
        raise ValueError("--tile-size must be larger than --tile-overlap")
    if tile_size % 2 or overlap % 2:
        raise ValueError("--tile-size and --tile-overlap must be even")

    height, width = mosaic.shape
    out_height = (height - (height % 2)) * args.scale_factor
    out_width = (width - (width % 2)) * args.scale_factor
    accum = np.zeros((out_height, out_width, 3), dtype=np.float32)
    weights = np.zeros((out_height, out_width, 1), dtype=np.float32)
    stride = tile_size - overlap
    y_starts = list(range(0, max(height - overlap, 1), stride))
    x_starts = list(range(0, max(width - overlap, 1), stride))
    y_starts[-1] = max(0, height - tile_size)
    x_starts[-1] = max(0, width - tile_size)
    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            if (y1 - y0) % 2:
                y1 -= 1
            if (x1 - x0) % 2:
                x1 -= 1
            tile = mosaic[y0:y1, x0:x1]
            pred = predict_array(model, tile, args.noise, max_value, add_noise)
            oy0 = y0 * args.scale_factor
            ox0 = x0 * args.scale_factor
            oy1 = oy0 + pred.shape[0]
            ox1 = ox0 + pred.shape[1]
            window = blend_window(pred.shape[0], pred.shape[1])
            accum[oy0:oy1, ox0:ox1] += pred * window
            weights[oy0:oy1, ox0:ox1] += window
            print("Tile y={}..{} x={}..{} done".format(y0, y1, x0, x1))

    return accum / np.maximum(weights, 1.0e-6)


def remosaic_rgb(rgb, pattern, max_value):
    height, width, _ = rgb.shape
    out = np.empty((height, width), dtype=np.float32)
    order = pattern.lower()
    for y_parity in range(2):
        for x_parity in range(2):
            color = order[y_parity * 2 + x_parity]
            out[y_parity::2, x_parity::2] = rgb[y_parity::2, x_parity::2, COLOR_INDEX[color]]
    return np.clip(np.rint(out * max_value), 0, max_value).astype(np.uint16)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_pattern = args.output_pattern or args.pattern
    max_value = (1 << args.bit_depth) - 1
    add_noise = args.noise > 0

    raw = read_raw(args.input, args.width, args.height, args.byte_order)
    if raw.max() > max_value:
        print("Warning: input contains values above {} for {}-bit data.".format(max_value, args.bit_depth))

    network_raw = align_to_rggb(raw, args.pattern)

    model = make_model(args, add_noise)
    rgb = predict_tiled(model, network_raw, args, max_value, add_noise)
    rgb = undo_alignment(rgb, args.pattern, args.scale_factor)

    base = args.output_name or os.path.splitext(os.path.basename(args.input))[0]
    tiff_path = os.path.join(args.output_dir, base + "_jdndmsr_rgb16.tiff")
    png_path = os.path.join(args.output_dir, base + "_jdndmsr_preview.png")
    raw_path = os.path.join(args.output_dir, base + "_jdndmsr_bayer{}.raw".format(output_pattern.upper()))
    rgb16 = np.clip(np.rint(rgb * max_value), 0, max_value).astype(np.uint16)
    cv2.imwrite(tiff_path, cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR))
    cv2.imwrite(png_path, cv2.cvtColor((rgb * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    raw_out = remosaic_rgb(rgb, output_pattern, max_value)
    write_raw(raw_path, raw_out, args.byte_order)
    print("Wrote viewable preview: {}".format(png_path))
    print("Wrote viewable RGB: {}".format(tiff_path))
    print("Wrote remosaiced RAW: {}".format(raw_path))
    print("Output size: {}x{} pixels".format(rgb.shape[1], rgb.shape[0]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise
