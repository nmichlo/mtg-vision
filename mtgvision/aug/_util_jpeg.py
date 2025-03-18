import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.fft import dct, idct
from jax.image import resize

# Base quantization tables (standard JPEG)
_q_luminance_base = jnp.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

_q_chrominance_base = jnp.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


# Color space conversion
def _rgb_to_ycbcr(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 + (-0.1687 * r - 0.3313 * g + 0.5 * b)
    cr = 128 + (0.5 * r - 0.4187 * g - 0.0813 * b)
    return jnp.stack([y, cb, cr], axis=-1)


def _ycbcr_to_rgb(ycbcr):
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    r = y + 1.402 * (cr - 128)
    g = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return jnp.stack([r, g, b], axis=-1)


# DCT/IDCT
def _dct2d(x):
    return dct(dct(x, axis=2, norm="ortho"), axis=3, norm="ortho")


def _idct2d(x):
    return idct(idct(x, axis=2, norm="ortho"), axis=3, norm="ortho")


# Subsampling and upsampling
def _subsample(channel):
    H, W = channel.shape
    channel_down = channel.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))
    return channel_down


def _upsample(channel_down, target_shape):
    return resize(channel_down, target_shape, method="bilinear")


# Process a channel
def _process_channel(
    channel: jnp.ndarray, q_table: jnp.ndarray, subsample: bool = False
) -> jnp.ndarray:
    H, W = channel.shape
    if subsample:
        channel_down = _subsample(channel)
        H_down, W_down = channel_down.shape
        H_padded = ((H_down + 7) // 8) * 8
        W_padded = ((W_down + 7) // 8) * 8
        pad_H, pad_W = H_padded - H_down, W_padded - W_down
        channel_padded = jnp.pad(channel_down, ((0, pad_H), (0, pad_W)), mode="reflect")
    else:
        H_padded = ((H + 7) // 8) * 8
        W_padded = ((W + 7) // 8) * 8
        pad_H, pad_W = H_padded - H, W_padded - W
        channel_padded = jnp.pad(channel, ((0, pad_H), (0, pad_W)), mode="reflect")

    blocks = channel_padded.reshape(H_padded // 8, 8, W_padded // 8, 8).transpose(
        0, 2, 1, 3
    )
    blocks_shifted = blocks - 128
    dct_coeffs = _dct2d(blocks_shifted)
    q_table = q_table.astype(dct_coeffs.dtype)
    coeffs_quantized = jnp.round(dct_coeffs / q_table)
    print(f"Quality {quality}: coeffs_quantized mean = {jnp.mean(coeffs_quantized)}")
    coeffs_dequantized = coeffs_quantized * q_table
    blocks_reconstructed_shifted = _idct2d(coeffs_dequantized)
    blocks_reconstructed = blocks_reconstructed_shifted + 128
    channel_padded_reconstructed = blocks_reconstructed.transpose(0, 2, 1, 3).reshape(
        H_padded, W_padded
    )

    if subsample:
        channel_reconstructed_down = channel_padded_reconstructed[:H_down, :W_down]
        channel_reconstructed = _upsample(channel_reconstructed_down, (H, W))
    else:
        channel_reconstructed = channel_padded_reconstructed[:H, :W]
    return jnp.clip(channel_reconstructed, 0, 255) / 255


# Quantization scaling
def get_quantization_tables(quality):
    quality = max(1, min(quality, 100))
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    scale = max(1, scale)
    q_luminance = jnp.round((_q_luminance_base * scale + 50) / 100).clip(1, 255)
    q_chrominance = jnp.round((_q_chrominance_base * scale + 50) / 100).clip(1, 255)
    print(
        f"Quality {quality}: q_luminance[0,0] = {q_luminance[0, 0]}, q_chrominance[0,0] = {q_chrominance[0, 0]}"
    )
    return q_luminance, q_chrominance


# Main compression function
def rgb_img_jpeg_compression_jax(src: jnp.ndarray, quality: int) -> jnp.ndarray:
    ycbcr = _rgb_to_ycbcr(src * 255)
    q_luminance, q_chrominance = get_quantization_tables(quality)
    y = _process_channel(ycbcr[..., 0], q_luminance, subsample=False)
    cb = _process_channel(ycbcr[..., 1], q_chrominance, subsample=True)
    cr = _process_channel(ycbcr[..., 2], q_chrominance, subsample=True)
    ycbcr_reconstructed = jnp.stack([y, cb, cr], axis=-1)
    rgb_reconstructed = _ycbcr_to_rgb(ycbcr_reconstructed * 255)
    return jnp.clip(rgb_reconstructed, 0, 255) / 255


# OpenCV comparison
def _rgb_img_jpeg_compression_cv2(src, quality: int):
    img = np.clip(src * 255, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", img, encode_param)
    img_transformed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return img_transformed.astype(np.float32) / 255


# Test
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    img = jax.random.uniform(key, (64, 128, 3))
    for quality in [10, 50, 90]:
        print(f"\nQuality = {quality}")
        img_jax = rgb_img_jpeg_compression_jax(img, quality)
        img_cv2 = _rgb_img_jpeg_compression_cv2(img, quality)
        diff = jnp.abs(img_jax - img_cv2)
        plt.imshow(img_jax)
        plt.show()
        plt.imshow(img_cv2)
        plt.show()
        print(f"Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
