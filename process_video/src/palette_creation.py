import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm


def create_random_palette(n, sizeX, sizeY):
    """
    This function creates n random tiles of size sizeX x sizeY.
    """
    tiles = []
    for _ in range(n):
        tile = np.random.randint(0, 256, (sizeY, sizeX))
        tiles.append(tile)
    # blur the tiles
    for i in range(n):
        tiles[i] = gaussian_filter(tiles[i], sigma=2)

    # set 0 if value is less than 128
    np_tiles = np.array(tiles)
    np_tiles[np_tiles < 128] = 0
    np_tiles[np_tiles >= 128] = 255
    return np_tiles


def find_most_varied_pixel(tiles):
    # for each pixel, compute the variance of the tiles on that pixel
    var = np.zeros((tiles.shape[1], tiles.shape[2]))
    for i in range(tiles.shape[1]):
        for j in range(tiles.shape[2]):
            var[i, j] = np.var(tiles[:, i, j])
    # find the pixel with the highest variance
    pixel = np.unravel_index(np.argmax(var), var.shape)
    return pixel


def discriminate_on_pixel(tiles, pixel, indexes):
    # get mean of tiles in mask
    mean = np.mean(tiles[indexes, pixel[0], pixel[1]])
    # find list index of indexes where tiles[index, pixel[0], pixel[1]]
    pixel_values = tiles[indexes, pixel[0], pixel[1]]
    mask = pixel_values < mean
    indexes1 = indexes[mask]
    indexes2 = indexes[~mask]
    return indexes1, indexes2


def cut_tiles_for_n_bins(tiles, indexes, n_bins):
    # find the best pixel, discriminate the tiles on that pixel
    # then find the right ratio to split the tiles in n_bins, each tiles mask has its own n_bin

    pixel = find_most_varied_pixel(tiles[indexes])
    indexes1, indexes2 = discriminate_on_pixel(tiles, pixel, indexes)
    n_1 = indexes1.shape[0]
    n_2 = indexes2.shape[0]
    N = n_1 + n_2
    # b1 = min(max(n_1 * n_bins // N, 1), n_bins - 1)
    b1 = n_bins // 2
    b2 = n_bins - b1
    return (indexes1, b1), (indexes2, b2)


from src.dither import ordered_dithering, stucki


def get_single_palette(tiles, dithering_method="ordered"):
    # get the best palette "color" for the tiles
    # get the mean of the tiles
    # samples = samples - total_mean + 0.5
    mean = np.mean(tiles, axis=0).astype(np.int64)
    # print(mean)
    if dithering_method == "error":
        mean = stucki(mean)
    elif dithering_method == "ordered":
        mean = ordered_dithering(mean)
    elif dithering_method == "none":
        mean[mean > 127] = 255
        mean[mean <= 127] = 0
    else:
        raise ValueError("dithering_method should be 'error', 'ordered' or 'none'")
    # print(mean)
    return mean


def create_palette_from_tiles(tiles, n, dithering_method="ordered"):
    """
    This function takes a list of tiles and creates a palette of size n.
    """
    # find the best pixel, discriminate the tiles on that pixel
    # then find the right ratio to split the tiles in n_bins, each mask has its own n_bin
    # use a queue, update every time we find a palette

    index = np.arange(tiles.shape[0])
    res = []
    i = 1

    # already_found = set()
    palette = []

    # find all white tiles
    white_tiles_index = np.where(np.all(tiles == 255, axis=(1, 2)))[0]
    if white_tiles_index.shape[0] > 0:
        print(f"Found {white_tiles_index.shape[0]} pure white tiles")
        swatch = np.full((tiles.shape[1], tiles.shape[2]), 255)
        palette.append(swatch)
        # already_found.add(tuple(swatch))
        res.append(white_tiles_index)
        n -= 1

    # find all black tiles
    black_tiles_index = np.where(np.all(tiles == 0, axis=(1, 2)))[0]
    if black_tiles_index.shape[0] > 0:
        print(f"Found {black_tiles_index.shape[0]} pure black tiles")
        swatch = np.full((tiles.shape[1], tiles.shape[2]), 0)
        palette.append(swatch)
        # already_found.add(tuple(swatch))
        res.append(black_tiles_index)
        n -= 1

    index = np.where(
        np.logical_and(
            np.any(tiles != 0, axis=(1, 2)), np.any(tiles != 255, axis=(1, 2))
        )
    )[0]

    if white_tiles_index.shape[0] > 0 or black_tiles_index.shape[0] > 0:
        print(f"{index.shape[0]} tiles left")

    bins = [(index, n)]

    with tqdm(total=n) as pbar:
        while len(bins) > 0:
            (indexes, n_bins) = bins.pop(0)
            if indexes.shape[0] == 0:
                continue
            if n_bins == 0:
                raise ValueError("n_bins should be greater than 0")
            if n_bins == 1:
                swatch = get_single_palette(tiles[indexes], dithering_method)
                palette.append(swatch)
                # already_found.add(tuple(swatch))
                res.append(indexes)
                continue
            # print(f"[{i}] cutting {n_bins} bins,  {indexes.shape[0]} tiles")
            (indexes1, b1), (indexes2, b2) = cut_tiles_for_n_bins(
                tiles, indexes, n_bins
            )
            # print(
            #     f"[{i}] cut into {b1} and {b2} bins, "
            #     f"{indexes1.shape[0]} and {indexes2.shape[0]} tiles, "
            #     f"left {len(bins)} bins (+2)"
            # )
            i += 1
            pbar.update(1)
            bins.append((indexes1, b1))
            bins.append((indexes2, b2))

        pbar.update(1)

    print(f"Found {len(palette)} colors")
    # print(f"Found {len(already_found)} unique colors")

    palette = np.array(palette)
    closest_tile = np.zeros(tiles.shape[0]).astype(int)
    for i in range(len(palette)):
        for j in range(len(res[i])):
            closest_tile[res[i][j]] = i

    return palette, closest_tile
