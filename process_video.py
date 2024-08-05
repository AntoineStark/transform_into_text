#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
import argparse
import os

# Import your custom modules
from src.palette_creation import create_palette_from_tiles
from src.tiles import create_image_from_tiles, create_tiles_from_image
from src.img_manip import create_images_from_videos_and_resize, save_images_as_video
from src.save_to_file import save_palette_to_file, save_closest_tile_frames_to_file


def main(
    sizeX,
    sizeY,
    nX,
    nY,
    n_palette,
    input_video,
    output_dir,
    no_video_output=False,
    dither="ordered",
    start=0,
    duration=-1,
):
    name = os.path.splitext(os.path.basename(input_video))[0]

    # create folder called f"{naem}_{n_palette}_{sizeY}" to store the output files

    if output_dir is not None:
        folder = output_dir
    else:
        folder = f"{name}_{n_palette}_{sizeY}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    video_name = f"{folder}/{name}_{n_palette}_{sizeY}_dithered.mp4"
    palette_name = f"{folder}/{name}_{n_palette}_{sizeY}_palette.txt"
    closest_tile_name = f"{folder}/{name}_{n_palette}_{sizeY}_closest_tile.txt"

    cap = cv.VideoCapture(input_video)
    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    print(f"Loading video {input_video} with {frame_count} frames at {fps} fps...")
    if duration > 0:
        print(f"Processing from {start} to {start + duration} seconds...")
    frames = create_images_from_videos_and_resize(
        cap, sizeX, sizeY, nX, nY, start, duration
    )
    print(f"{len(frames)} frames loaded.")

    print()

    print(f"Creating {sizeX}x{sizeY} tiles ({nX}x{nY} tiles per frame)...")
    tiless = np.array(
        [create_tiles_from_image(img, sizeX, sizeY) for img in tqdm(frames)]
    )
    tiles = np.concatenate(tiless, axis=0)
    print(f"{len(tiles)} tiles created.")

    print()

    print(f"Creating palette of {n_palette} tiles...")
    palette, closest_tile = create_palette_from_tiles(tiles, n_palette, dither)
    closest_tile_frames = closest_tile.reshape(tiless.shape[0], tiless.shape[1])
    print(f"palette of {len(palette)} created.")

    print()

    print(f"Saving palette as {palette_name}...")
    save_palette_to_file(palette, n_palette, palette_name)
    print(f"{palette_name} saved.")

    print()

    print(f"Saving closest tile frames as {closest_tile_name}...")
    save_closest_tile_frames_to_file(closest_tile_frames, closest_tile_name, nX, nY)
    print(f"{closest_tile_name} saved.")

    print()

    if not no_video_output:
        print("Creating dithered frames...")
        frames_dith = []
        for i in trange(tiless.shape[0]):
            closest_tile_frame = closest_tile_frames[i]
            frame = create_image_from_tiles(palette[closest_tile_frame], nX, nY)

            frames_dith.append(frame)
        print(f"{len(frames_dith)} dithered frames created.")

        print()

        print("Saving dithered video...")
        save_images_as_video(frames_dith, video_name, fps)
        print(f"{video_name} saved.")
    else:
        print("Video output not saved per user request.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process video and create a dithered output."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input video file path, required"
    )
    parser.add_argument("--start", type=int, default=0, help="Start second, default 0")
    parser.add_argument(
        "--duration", type=int, default=-1, help="Duration in seconds, default -1 (all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Input video file path, default is input video name",
    )
    parser.add_argument(
        "--sizeX", type=int, default=8, help="Tile width in pixels, default 8"
    )
    parser.add_argument(
        "--sizeY", type=int, default=8, help="Tile height in pixels, default 8"
    )
    parser.add_argument(
        "--nX", type=int, default=90, help="Number of tiles horizontally, default 90"
    )
    parser.add_argument(
        "--nY", type=int, default=43, help="Number of tiles vertically, default 43"
    )
    parser.add_argument(
        "--dither",
        type=str,
        default="ordered",
        help="Dithering method, can choose between ordered, error and none, default ordered",
    )
    parser.add_argument(
        "--n_palette",
        type=int,
        default=3072,
        help="Number of tiles in the palette, default 3072",
    )
    parser.add_argument(
        "--no-video-output",
        action="store_true",
        help="Do not save the output video, default False",
    )

    args = parser.parse_args()

    print(f"Arguments: {args}")

    main(
        args.sizeX,
        args.sizeY,
        args.nX,
        args.nY,
        args.n_palette,
        args.input,
        args.output,
        args.no_video_output,
        args.dither,
        args.start,
        args.duration,
    )
