import numpy as np
from tqdm.auto import trange


def save_palette_to_file(palette, n, filename):
    """
    This function saves the palette to a file.
    """

    length = len(palette)
    palette = np.array(palette)
    palette = palette.copy()
    if length < n:
        palette = np.concatenate(
            (palette, np.zeros((n - length, palette.shape[1], palette.shape[2])))
        )

    with open(filename, "w") as f:
        f.write(f"FONT\n\n{length}\n\n")
        for swatch in palette:
            for i in range(8):
                swatch_line = (swatch[i] > 128).astype(int)
                f.write("".join([str(x) for x in swatch_line]) + "\n")
            f.write("\n")


def save_closest_tile_frames_to_file(closest_tile_frames, filename, nX, nY):
    """
    This function saves the indexes to a file.
    """
    with open(filename, "w") as f:
        f.write(f"FONT\n\n{nX} {nY}\n\n")
        for i in trange(closest_tile_frames.shape[0]):
            closest_tile_frame = closest_tile_frames[i]
            for j in range(closest_tile_frame.shape[0]):
                f.write(str(closest_tile_frame[j]) + " ")
            f.write("\n")
