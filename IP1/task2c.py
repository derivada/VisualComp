# Code for task 2, exercise c
# Pablo Díaz Viñambres

import matplotlib.pyplot as plt
import pathlib
import numpy as np
import time
from utils import read_im, save_im, normalize
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)


def convolve_im(im, kernel,
                ):
    """ A function that convolves im with kernel

    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]

    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    start = time.perf_counter() # Performance counter for testing optimizations
    # Convolve for each color channel
    for ch in range(3):
        # We make another matrix since the overlap of the kernel would alter the results 
        tmp_convolve = np.zeros(im.shape[0:2])
        row_margin = int((kernel.shape[0] - 1) / 2)
        col_margin = int((kernel.shape[1] - 1) / 2)
        for i in range(row_margin, im.shape[0] - row_margin):
            for j in range(col_margin, im.shape[1] - col_margin): 
                # Numpy operators make the operation faster, we extract the correct square from the original values matrix 
                # and multiply it element wise with the kernel 
                tmp_convolve[i, j] = np.sum(kernel[:, :] * im[i - row_margin : i + row_margin + 1, j - col_margin : j + col_margin + 1, ch])
                # Slower version without numpy operators
                # for k in range(kernel.shape[0]):
                #     for l in range(kernel.shape[1]):
                #         tmp_convolve[i, j] += kernel[k, l] * im[i - k, j - l, ch]
        # Copy back to image channel
        im[:, :, ch] = tmp_convolve
    stop = time.perf_counter()
    print('time elapsed: {}'.format(stop-start))
    assert len(im.shape) == 3
    return im


if __name__ == "__main__":
    # Define the convolutional kernels
    h_b = 1 / 256 * np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Convolve images
    im_smoothed = convolve_im(im.copy(), h_b)
    save_im(output_dir.joinpath("im_smoothed.jpg"), im_smoothed)
    im_sobel = convolve_im(im, sobel_x)
    save_im(output_dir.joinpath("im_sobel.jpg"), im_sobel)

    # DO NOT CHANGE. Checking that your function returns as expected
    assert isinstance(
        im_smoothed, np.ndarray),         f"Your convolve function has to return a np.array. " + f"Was: {type(im_smoothed)}"
    assert im_smoothed.shape == im.shape,         f"Expected smoothed im ({im_smoothed.shape}" + \
        f"to have same shape as im ({im.shape})"
    assert im_sobel.shape == im.shape,         f"Expected smoothed im ({im_sobel.shape}" + \
        f"to have same shape as im ({im.shape})"
    plt.subplot(1, 2, 1)
    plt.imshow(normalize(im_smoothed))

    plt.subplot(1, 2, 2)
    plt.imshow(normalize(im_sobel))
    plt.show()
