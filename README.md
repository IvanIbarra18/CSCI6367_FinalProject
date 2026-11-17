# Digital Image Processing Final Project

This project analyzes a grayscale lung CT image in the spatial and frequency domains. It uses a two-dimensional Fast Fourier Transform (2D FFT), downsamples the image by half, reconstructs it with two interpolation methods, and compares the reconstructions with the original.

## Requirements

- Python 3
- Packages listed in [`requirements.txt`](requirements.txt)

## Setup

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Run

Run the analysis on the lung CT image:

```powershell
python main.py
```

The default image is `lung_ct`. To select the other included image, use:

```powershell
python main.py --image test
```

The script displays each figure as it is generated. To also save the figures as PNG files under `results/<image>/`, add `--save-plots`:

```powershell
python main.py --image lung_ct --save-plots
```

The available image names are `lung_ct` and `test`; the corresponding JPEG files are in [`imgs/`](imgs/).

## Processing steps

For the default lung CT image, the script:

1. Loads the image in grayscale and reports its dimensions (426 × 640 pixels).
2. Computes the 2D FFT and displays its magnitude and phase, both before and after centering the spectrum with `fftshift`. It also displays a log-scaled shifted magnitude.
3. Downsamples the image to 213 × 320 by taking every other row and column.
4. Reconstructs an image at the original size by zero-padding the centered downsampled spectrum and applying the inverse 2D FFT. The padding handles the even-sized horizontal Nyquist component to preserve conjugate symmetry, and scales the result to account for the inverse FFT normalization.
5. Reconstructs an image using bilinear interpolation in the spatial domain.
6. Compares both reconstructions with the original using normalized squared error.

## Reported results

The normalized squared errors reported for `lung_ct` are:

| Interpolation method | Normalized squared error |
| --- | ---: |
| Frequency domain (zero padding) | 0.00109284 |
| Spatial domain (bilinear) | 0.00077360 |

The bilinear reconstruction had the lower error for this image. These values are specific to the included lung CT image and the implementation; interpolation does not recover detail lost during downsampling.

The metric used is the normalized squared error specified in the assignment:

```text
sum((original - interpolated)^2) / sum(original^2)
```

Although the assignment calls this metric MSE, it is normalized by the original image's total squared intensity rather than by the number of pixels.
