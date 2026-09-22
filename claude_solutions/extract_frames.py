# Pull sharp frames out of the video and save them as images
# PIV - PBClass 2
# Run this first, then run calibration.py
# Run:  python3 extract_frames.py

import os
import sys

import cv2


# where this script lives, so the paths work from any folder you run it in
HERE = os.path.dirname(os.path.abspath(__file__))

#VIDEO_FILE = os.path.join(HERE, "..", "prob", "lego_twogrids.mp4")
VIDEO_FILE = os.path.join(HERE, "..", "piv", "video_s21_pedro.mp4")
FRAMES_DIR = os.path.join(HERE, "frames")

# take one frame out of every FRAME_STEP
FRAME_STEP = 10

# blurry frames give bad corners, so skip them
MIN_SHARPNESS = 100.0

MAX_FRAMES = 30


# ---- make the folder, and empty it if it already had frames ----

os.makedirs(FRAMES_DIR, exist_ok=True)

old_names = os.listdir(FRAMES_DIR)
removed = 0

for name in old_names:
    if not name.endswith(".png"):
        continue

    path = os.path.join(FRAMES_DIR, name)
    os.remove(path)
    removed = removed + 1

if removed > 0:
    print("removed", removed, "old frames")


# ---- walk through the video ----

video = cv2.VideoCapture(VIDEO_FILE)

if not video.isOpened():
    print("ERROR: could not open", VIDEO_FILE)
    sys.exit()

saved = 0
skipped_blurry = 0
frame_number = -1

while True:
    got_one, frame = video.read()

    if not got_one:
        break

    frame_number = frame_number + 1

    if frame_number % FRAME_STEP != 0:
        continue

    if saved >= MAX_FRAMES:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # a blurry picture has soft edges, so the Laplacian barely changes
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    if sharpness < MIN_SHARPNESS:
        skipped_blurry = skipped_blurry + 1
        continue

    name = "frame_%03d.png" % frame_number
    path = os.path.join(FRAMES_DIR, name)
    cv2.imwrite(path, frame)

    saved = saved + 1

video.release()

print("saved", saved, "frames into", FRAMES_DIR)
print("skipped", skipped_blurry, "blurry frames")
