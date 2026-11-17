import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

def parse_args():
    parser = argparse.ArgumentParser(description="Digital Image Processing - Final Project")
    parser.add_argument("--image",type=str,choices=["lung_ct", "test"],default="lung_ct",help="Select the input image: lung_ct or test")
    parser.add_argument("--save-plots",action="store_true",help="Save every displayed figure as a PNG in results/<image>/")
    return parser.parse_args()

def show_plot(name):
    if args.save_plots:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / f"{name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
    plt.show()

def normalized_mse(original, interp):

    if original.shape != interp.shape:
        raise ValueError("Images must have the same dimensions")

    original = original.astype(np.float64)
    interp = interp.astype(np.float64)

    error = np.sum((original - interp) ** 2)
    norm = np.sum(original ** 2)

    if norm == 0:
        raise ValueError("Original image has zero energy")

    return error / norm

def zero_pad_spectrum(F_shifted, output_shape):

    m, n = F_shifted.shape
    M, N = output_shape

    padded = np.zeros((M, N), dtype=np.complex128)

    # Top-left corner of the m x n block so that both DC terms line up
    r0 = M // 2 - m // 2
    c0 = N // 2 - n // 2

    padded[r0:r0 + m, c0:c0 + n] = F_shifted

    # Split the Nyquist row (-m/2 -> -m/2 and +m/2)
    if m % 2 == 0 and M > m:
        padded[r0, :] *= 0.5
        padded[r0 + m, :] = padded[r0, :]

    # Split the Nyquist column (-n/2 -> -n/2 and +n/2)
    if n % 2 == 0 and N > n:
        padded[:, c0] *= 0.5
        padded[:, c0 + n] = padded[:, c0]

    return padded


args = parse_args()

# Get the directory containing this script
BASE_DIR = Path(__file__).resolve().parent

# Select the input image
image_path = BASE_DIR / "imgs" / f"{args.image}.jpg"

# Output folder for saved figures (used with --save-plots)
RESULTS_DIR = BASE_DIR / "results" / args.image


# (1). Load and display this grey-scale lung CT image.

# Load image in grayscale
img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

# Check if loaded correctly
if img is None:
    raise FileNotFoundError(f"Error loading image: {image_path}")

print(f"Image loaded successfully: {args.image}.jpg")

# Report the size of this image (M×N).
M, N = img.shape

print(f"Image size: {M} x {N}")

# Display image
plt.figure(figsize=(8, 6))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Lung CT Image')
plt.axis('off')
plt.tight_layout()
show_plot("01_original")

# (2). Compute the 2D FFT and display

FFT = np.fft.fft2(img.astype(np.float64))

# (a) magnitude

magnitude = np.abs(FFT)

plt.figure(figsize=(8, 6))
plt.imshow(magnitude, cmap='gray')
plt.title('Magnitude of FFT')
plt.colorbar(label='Magnitude')
plt.tight_layout()
show_plot("02_fft_magnitude")

# (b) phase

phase = np.angle(FFT)

plt.figure(figsize=(8, 6))
plt.imshow(phase, cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.title('Phase of FFT')
plt.colorbar(label='Phase (radians)')
plt.tight_layout()
show_plot("03_fft_phase")

# Now apply fftshift and again display

FFT_shift = np.fft.fftshift(FFT)

# (c) magnitude

shift_Magnitude = np.abs(FFT_shift)

plt.figure(figsize=(8, 6))
plt.imshow(shift_Magnitude, cmap='gray')
plt.title('Magnitude of FFT (Shifted)')
plt.colorbar(label='Magnitude')
plt.tight_layout()
show_plot("04_fft_magnitude_shifted")

# (d) phase

shift_Phase = np.angle(FFT_shift)

plt.figure(figsize=(8, 6))
plt.imshow(
    shift_Phase,
    cmap='gray',
    vmin=-np.pi,
    vmax=np.pi
)
plt.title('Phase of FFT (Shifted)')
plt.colorbar(label='Phase (radians)')
plt.tight_layout()
show_plot("05_fft_phase_shifted")

# Use a log-scale for the magnitude display if this produces a more informative results.

log_magnitude = np.log1p(shift_Magnitude)

plt.figure(figsize=(8, 6))
plt.imshow(log_magnitude, cmap='gray')
plt.title('Log Magnitude of FFT (Shifted)')
plt.colorbar(label='Log Magnitude')
plt.tight_layout()
show_plot("06_fft_log_magnitude_shifted")


# (3). Down-sample the image to a half-size (M/2×N/2)
# image by discarding odd-numbered samples and display it.

# Keep zero-based even-indexed samples
img_down = img[::2, ::2]

M2, N2 = img_down.shape

print(f"Downsampled size: {M2} x {N2}")

plt.figure(figsize=(8, 6))
plt.imshow(img_down, cmap='gray', vmin=0, vmax=255)
plt.title('Downsampled Image (M/2 x N/2)')
plt.axis('off')
plt.tight_layout()
show_plot("07_downsampled")


# (4). Get the frequency domain of the down-sampled image
# (Do 2D FFT on (3)).

FFT_down = np.fft.fft2(img_down.astype(np.float64))

# Shift
FFT_down_shifted = np.fft.fftshift(FFT_down)

# Then you write a program to interpolate the FFT result
# in the frequency domain to make it back to the original size (M × N)
# by utilizing zeropadding method.

# Specifically, you need to FFT the down-sampled image,
# then zeropad in the frequency domain to interpolate,
# and then inverse 2D FFT.

# You need to figure out where the zero should be padded in so that you can
# retain conjugate symmetry in frequencies so that the resulting image is real.
# Display the result after inverse 2D FFT.

# (Conjugate symmetry: We have shown in class that for a real
# image, the frequency should satisfy:
# F(u, v) = F*(−u,−v).
# * means conjugate).

# Display the centered spectrum of the down-sampled image
plt.figure(figsize=(8, 6))
plt.imshow(np.log1p(np.abs(FFT_down_shifted)), cmap='gray')
plt.title('Log Magnitude of FFT of Downsampled Image (Shifted)')
plt.colorbar(label='Log Magnitude')
plt.tight_layout()
show_plot("08_fft_downsampled_shifted")

# Zero-pad the centered spectrum from (M/2 x N/2) to (M x N).
# The original frequencies go in the center; zeros fill the border
# (the new high frequencies). Nyquist rows/columns are split in half
# so that F(u, v) = F*(-u, -v) still holds.
FFT_padded_shifted = zero_pad_spectrum(FFT_down_shifted, (M, N))


# Verify conjugate symmetry: F(u, v) = F*(-u, -v).
# In unshifted indexing, -u corresponds to (-u mod M).
FFT_padded = np.fft.ifftshift(FFT_padded_shifted)
FFT_mirror = np.roll(FFT_padded[::-1, ::-1], shift=(1, 1), axis=(0, 1))
symmetry_error = np.max(np.abs(FFT_padded - np.conj(FFT_mirror)))

print(f"Conjugate symmetry error: {symmetry_error:.12e}")

# Inverse 2D FFT before amplitude scaling.
img_interp_freq_unscaled_complex = np.fft.ifft2(FFT_padded)
img_interp_freq_unscaled = np.real(img_interp_freq_unscaled_complex)

plt.figure(figsize=(8, 6))
plt.imshow(img_interp_freq_unscaled, cmap='gray')
plt.title('IFFT Result Before Scaling Factor')
plt.axis('off')
plt.tight_layout()
show_plot("9_ifft_before_scaling")

# Scale by (M*N)/(m*n) because numpy's ifft2 divides by M*N
# while the spectrum came from an m*n FFT.
scale = (M * N) / (M2 * N2)
img_interp_freq_complex = img_interp_freq_unscaled_complex * scale

# Check the imaginary component before taking the real part
imaginary_error = np.max(np.abs(np.imag(img_interp_freq_complex)))

print(f"Maximum imaginary component: {imaginary_error:.12e}")

# Inverse FFT result should be real for a real-valued input image
img_interp_freq = np.real(img_interp_freq_complex)

plt.figure(figsize=(8, 6))
plt.imshow(img_interp_freq, cmap='gray', vmin=0, vmax=255)
plt.title('Interpolated Image (Frequency Domain)')
plt.axis('off')
plt.tight_layout()
show_plot("10_interp_frequency_domain")

print(
    f"Image size frequency interpolated: "
    f"{img_interp_freq.shape[0]} x {img_interp_freq.shape[1]}"
)

# (5). Now you perform a spatial domain interpolation.
# Find a function to perform a linear interpolation
# of your image from the part (3). Display it.

# Convert image into rows and columns
rows, cols = img_down.shape

# Coordinates of the original samples retained during downsampling
x = np.arange(cols) * 2
y = np.arange(rows) * 2

# Create function of linear interpolation
linear_interp_func = RegularGridInterpolator(
    (y, x),
    img_down.astype(np.float64),
    method='linear',
    bounds_error=False,
    fill_value=None
)

# Convert back into an image
# Generate coordinates in the original image domain
yy, xx = np.meshgrid(
    np.arange(M),
    np.arange(N),
    indexing='ij'
)

# Clamp coordinates at the image boundaries
# to avoid extrapolation beyond the last available sample
yy = np.clip(yy, y[0], y[-1])
xx = np.clip(xx, x[0], x[-1])

points = np.column_stack((yy.ravel(), xx.ravel()))

img_interp_spatdom = linear_interp_func(points).reshape(M, N)

# Display :D
plt.figure(figsize=(8, 6))
plt.imshow(img_interp_spatdom, cmap='gray', vmin=0, vmax=255)
plt.title('Linearly Interpolated Image (Spatial Domain)')
plt.axis('off')
plt.tight_layout()
show_plot("11_interp_spatial_domain")

print(
    f"Image size spatially interpolated: "
    f"{img_interp_spatdom.shape[0]} x {img_interp_spatdom.shape[1]}"
)

# (6). Compute the errors between the original image (1)
# and your two interpolation results (4) and (5) using the mean squared error.

# Compute normalized squared errors using the professor's formula
normalized_mse_freq = normalized_mse(img, img_interp_freq)
normalized_mse_spatdom = normalized_mse(img, img_interp_spatdom)

print("\nNormalized Squared Error:")
print(f"Frequency Domain: {normalized_mse_freq:.8f}")
print(f"Spatial Domain:   {normalized_mse_spatdom:.8f}")