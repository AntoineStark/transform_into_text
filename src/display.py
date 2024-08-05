import matplotlib.pyplot as plt
import numpy as np


def display_image(img, ax=None):
    """
    This function takes an image and plots it using matplotlib.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.axis("off")


def display_images(images, plt=plt, n_max=16, randomize=True):
    """
    This function takes a list of images and plots them using matplotlib.
    """
    # display images in a gid of sqrt(len(images)) x sqrt(len(images))
    n = len(images)
    if randomize:
        images = np.random.permutation(images)
    if n > n_max:
        images = images[:n_max]
        n = n_max
    rows = int(n**0.5)
    cols = n // rows
    if rows * cols < n:
        cols += 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        if i < n:
            display_image(images[i], ax=ax)
        else:
            # display a cross
            ax.plot([0, 1], [0, 1], color="black")
            ax.plot([1, 0], [0, 1], color="black")
            ax.axis("off")

    # bring plots closer together
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
