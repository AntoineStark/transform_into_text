import numpy as np


def dither_matrix(n: int):
    if n == 1:
        return np.array([[0]])
    else:
        first = (n**2) * dither_matrix(int(n / 2))
        second = (n**2) * dither_matrix(int(n / 2)) + 2
        third = (n**2) * dither_matrix(int(n / 2)) + 3
        fourth = (n**2) * dither_matrix(int(n / 2)) + 1
        first_col = np.concatenate((first, third), axis=0)
        second_col = np.concatenate((second, fourth), axis=0)
        return (1 / n**2) * np.concatenate((first_col, second_col), axis=1)


dither_m = dither_matrix(8) * 255


def ordered_dithering(img_pixel: np.array):
    img_pixel = img_pixel.copy()
    n = np.size(dither_m, axis=0)
    x_max = np.size(img_pixel, axis=1)
    y_max = np.size(img_pixel, axis=0)
    for x in range(x_max):
        for y in range(y_max):
            i = x % n
            j = y % n
            if img_pixel[y][x] > dither_m[i][j]:
                img_pixel[y][x] = 255
            else:
                img_pixel[y][x] = 0
    return img_pixel


def set_pixel(im, x, y, new):
    im[x, y] = new


def stucki(im):  # stucki algorithm for image dithering
    w8 = 8 / 42.0
    w7 = 7 / 42.0
    w5 = 5 / 42.0
    w4 = 4 / 42.0
    w2 = 2 / 42.0
    w1 = 1 / 42.0
    width, height = im.shape
    for y in range(0, height - 2):
        for x in range(0, width - 2):
            old_pixel = im[x, y]
            if old_pixel < 127:
                new_pixel = 0
            else:
                new_pixel = 255
            set_pixel(im, x, y, new_pixel)
            quant_err = old_pixel - new_pixel
            set_pixel(im, x + 1, y, im[x + 1, y] + w7 * quant_err)
            set_pixel(im, x + 2, y, im[x + 2, y] + w5 * quant_err)
            set_pixel(im, x - 2, y + 1, im[x - 2, y + 1] + w2 * quant_err)
            set_pixel(im, x - 1, y + 1, im[x - 1, y + 1] + w4 * quant_err)
            set_pixel(im, x, y + 1, im[x, y + 1] + w8 * quant_err)
            set_pixel(im, x + 1, y + 1, im[x + 1, y + 1] + w4 * quant_err)
            set_pixel(im, x + 2, y + 1, im[x + 2, y + 1] + w2 * quant_err)
            set_pixel(im, x - 2, y + 2, im[x - 2, y + 2] + w1 * quant_err)
            set_pixel(im, x - 1, y + 2, im[x - 1, y + 2] + w2 * quant_err)
            set_pixel(im, x, y + 2, im[x, y + 2] + w4 * quant_err)
            set_pixel(im, x + 1, y + 2, im[x + 1, y + 2] + w2 * quant_err)
            set_pixel(im, x + 2, y + 2, im[x + 2, y + 2] + w1 * quant_err)

    im[im > 127] = 255
    im[im <= 127] = 0
    return im
