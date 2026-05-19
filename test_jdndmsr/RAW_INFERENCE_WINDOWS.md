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

Set `--tile-size 0` to run the whole frame at once. If the GPU runs out of
memory, use a smaller even tile size such as `768` or `512`.
