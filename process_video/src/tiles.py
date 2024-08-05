import numpy as np


def create_tiles_from_image(img, sizeX, sizeY):
    """
    This function takes an image and creates a list of tiles from it.
    The tiles are of size sizeX x sizeY.
    """
    # check that the image is divisible by the size of the tiles
    assert img.shape[0] % sizeX == 0
    assert img.shape[1] % sizeY == 0
    # check that the image is grayscale
    assert len(img.shape) == 2

    img = np.copy(img).astype(np.float32)
    # contrast image, bring 5% to 255 and 5% to 0
    img = (img - 128) * 1.1 + 128
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)

    tiles = []
    nX = img.shape[1] // sizeX
    nY = img.shape[0] // sizeY
    for i in range(nY):
        for j in range(nX):
            tiles.append(img[i * sizeY : (i + 1) * sizeY, j * sizeX : (j + 1) * sizeX])
    return np.array(tiles)


def create_image_from_tiles(tiles, nX, nY):
    """
    This function takes a list of tiles and reconstructs the image.
    """
    # check that the number of tiles is correct
    assert len(tiles) == nX * nY

    sizeY, sizeX = tiles[0].shape
    img = np.zeros((nY * sizeY, nX * sizeX))
    for i in range(nY):
        for j in range(nX):
            img[i * sizeY : (i + 1) * sizeY, j * sizeX : (j + 1) * sizeX] = tiles[
                i * nX + j
            ]
    return img
