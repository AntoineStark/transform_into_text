import cv2 as cv
import numpy as np

from tqdm.auto import tqdm


def resize_image(img, sizeX, sizeY, nX, nY):
    # crops image to be a w_target x h_target image
    h, w = img.shape[:2]

    w_target = sizeX * nX
    h_target = sizeY * nY
    # resize image to 512x512

    img = cv.resize(img, (w_target, h_target), interpolation=cv.INTER_CUBIC)

    return img


def create_images_from_videos_and_resize(cap, sizeX, sizeY, nX, nY, start, duration):
    """
    This function takes a video and creates a list of tiles from it.
    The tiles are of size sizeX x sizeY.
    """
    frames = []
    cap.set(cv.CAP_PROP_POS_MSEC, start * 1000)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv.CAP_PROP_POS_MSEC) > (start + duration) * 1000:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = resize_image(frame, sizeX, sizeY, nX, nY)
        frames.append(frame)

    return frames


def save_images_as_video(images, name, framerate=24):
    images = np.array(images).astype(np.uint8)
    # save images as a 30fps mp4 video
    # Retrieve the dimensions of the first image
    height, width = images[0].shape[:2]

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*"avc1")  # 'mp4v' for .mp4
    out = cv.VideoWriter(name, fourcc, framerate, (width, height), isColor=False)

    # Write each image to the video
    for image in tqdm(images):
        out.write(image)

    # Release everything when the job is finished
    out.release()
