import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter
from tqdm.auto import trange


def closest_palet_to_tiles(tiles, palette):
    """
    This function applies the palette to the samples.
    """
    # create a copy of the samples
    tiles = tiles.copy()
    palette = palette.copy()
    # blur the samples
    for i in range(len(tiles)):
        tiles[i] = gaussian_filter(tiles[i], sigma=2)

    # blur the palette
    for i in range(len(palette)):
        palette[i] = gaussian_filter(palette[i], sigma=2)

    dist = np.zeros((len(tiles), len(palette)))
    for i in trange(len(palette)):
        diff = tiles - palette[i]
        norm = np.linalg.norm(diff, axis=(1, 2))
        dist[:, i] = norm

    # find the closest tile for each sample
    closest_tile = np.argmin(dist, axis=1)
    return closest_tile
