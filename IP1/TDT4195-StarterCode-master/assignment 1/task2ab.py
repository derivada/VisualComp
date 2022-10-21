import matplotlib.pyplot as plt
import pathlib
from utils import read_im, save_im
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)


def greyscale(im):
    """ Converts an RGB image to greyscale

    Args:
        im ([type]): [np.array of shape [H, W, 3]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    # We apply the weighted sum gray_ij = 0.212 * red_ij + 0.7152 * green_ij + 0.0722 * blue_ij to the image
    im = 0.212 * im[:, :, 0] + 0.7152 * im[:, :, 1] + 0.0722 * im[:, :, 2]
    return im


im_greyscale = greyscale(im)
save_im(output_dir.joinpath("lake_greyscale.jpg"), im_greyscale, cmap="gray")
plt.imshow(im_greyscale, cmap="gray")


def inverse(im):
    """ Finds the inverse of the greyscale image

    Args:
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """
    # For the inverse, we simply compute the inverse of every pixel on the image
    im = 1 - im
    return im

im_grayscale_inverted = inverse(im_greyscale)
save_im(output_dir.joinpath("lake_greyscale_inv.jpg"), im_grayscale_inverted, cmap="gray")
plt.imshow(im_grayscale_inverted, cmap="gray")
