# Single RAW Inference on Windows

This repository's trained models are Keras 2.3.1 / TensorFlow GPU 1.14 weights.
For GPU inference on Windows, use a legacy Python/CUDA stack:

- Python 3.6 is the safest choice for `tensorflow-gpu==1.14.0`.
- CUDA 10.0 and cuDNN 7.x must be installed and visible on `PATH`.
- Conda is not required; `setup_venv_windows.bat` creates a local venv.

## Setup

```bat
setup_venv_windows.bat
.venv-jdndmsr\Scripts\activate.bat
```

## Bayer RAW Example

Input is assumed to be a headerless uint16 RAW file containing 10-bit values
stored in 16-bit little-endian words. The default pixel order is GRBG.

```bat
python infer_raw.py ^
  --input C:\data\frame.raw ^
  --width 4000 ^
  --height 3000 ^
  --pattern grbg ^
  --model models\jdndmsr+_model.h5 ^
  --scale-factor 2 ^
  --noise 10 ^
  --tile-size 1024 ^
  --tile-overlap 64 ^
  --output-dir C:\data\jdndmsr_out
```

The script writes:

- `*_jdndmsr_preview.png`: directly viewable 8-bit RGB preview.
- `*_jdndmsr_rgb16.tiff`: 16-bit RGB output using the requested 10-bit range.
- `*_jdndmsr_bayerGRBG.raw`: remosaiced uint16 RAW output using the requested Bayer order.

## Quad-Bayer RAW Output

The model input is still regular Bayer. To write the output RAW as Quad Bayer,
use `--output-raw-mosaic quad-bayer`. The Quad Bayer output pixel order defaults
to GRBG unless you set `--output-pattern`.

```bat
python infer_raw.py ^
  --input C:\data\frame.raw ^
  --width 4000 ^
  --height 3000 ^
  --pattern grbg ^
  --output-raw-mosaic quad-bayer ^
  --model models\jdndmsr+_model.h5 ^
  --tile-size 1024 ^
  --output-dir C:\data\jdndmsr_out
```

This writes a RAW file named like `*_jdndmsr_quadbayerGRBG.raw`.

## Full RGB RAW Output

To write full-color RAW instead of a Bayer mosaic, use
`--output-raw-mosaic rgb`. The output is headerless uint16 data in
pixel-interleaved R-G-B order:

```text
R0 G0 B0 R1 G1 B1 R2 G2 B2 ...
```

```bat
python infer_raw.py ^
  --input C:\data\frame.raw ^
  --width 4000 ^
  --height 3000 ^
  --pattern grbg ^
  --output-raw-mosaic rgb ^
  --raw-output C:\data\jdndmsr_out\frame_rgb_interleaved.raw ^
  --model models\jdndmsr+_model.h5 ^
  --tile-size 1024 ^
  --output-dir C:\data\jdndmsr_out
```

Without `--raw-output`, this writes a RAW file named like
`*_jdndmsr_rgbRGB.raw`.

## Quad-Bayer RAW Input

The trained model only accepts regular Bayer input. If your input file is Quad
Bayer, pass `--input-raw-mosaic quad-bayer`. The script reads the full input
size, bins each same-color 2x2 Quad Bayer block to one regular Bayer sample,
and then feeds the binned Bayer image to the model.

With the default `--scale-factor 2`, a `4000x3000` Quad Bayer input becomes a
`2000x1500` Bayer tensor internally, and the model output returns to
`4000x3000`.

```bat
python infer_raw.py ^
  --input C:\data\quad_frame.raw ^
  --width 4000 ^
  --height 3000 ^
  --input-raw-mosaic quad-bayer ^
  --pattern grbg ^
  --output-raw-mosaic quad-bayer ^
  --model models\jdndmsr+_model.h5 ^
  --scale-factor 2 ^
  --tile-size 1024 ^
  --output-dir C:\data\jdndmsr_out
```

Set `--tile-size 0` to run the whole frame at once. If the GPU runs out of
memory, use a smaller even tile size such as `768` or `512`.
