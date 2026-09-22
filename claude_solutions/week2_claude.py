import os
import sys
import cv2
import numpy as np


FRAMES_DIR = "frames"

CELL_WIDTH = 15.8
CELL_HEIGHT = 9.6

# inner corners, not cells: 6x4 cells gives 5x3 inner corners
LEFT_COLS = 5
LEFT_ROWS = 3
RIGHT_COLS = 7
RIGHT_ROWS = 3

# the pattern starts one cell away from the seam
FIRST_CELL = 2

CHECK_IMAGE = "reprojection_check.png"

FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
STOP_RULE = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# =====================================================================
#  where the corners really are, in millimetres
# =====================================================================

# left wall: plane X = 0, and Y grows to the left in the picture
left_points = []

for row in range(LEFT_ROWS):
    for col in range(LEFT_COLS):
        cells = FIRST_CELL + (LEFT_COLS - 1 - col)
        y = cells * CELL_WIDTH
        z = (LEFT_ROWS - row) * CELL_HEIGHT
        left_points.append([0.0, y, z])

left_points = np.array(left_points, dtype=np.float32)

# right wall: plane Y = 0, and X grows to the right in the picture
right_points = []

for row in range(RIGHT_ROWS):
    for col in range(RIGHT_COLS):
        cells = FIRST_CELL + col
        x = cells * CELL_WIDTH
        z = (RIGHT_ROWS - row) * CELL_HEIGHT
        right_points.append([x, 0.0, z])

right_points = np.array(right_points, dtype=np.float32)

# both walls together, 36 points
board_points = np.vstack([left_points, right_points])

# the right wall on its own as a flat board, so Z = 0, for task 4.1
flat_points = []

for row in range(RIGHT_ROWS):
    for col in range(RIGHT_COLS):
        x = col * CELL_WIDTH
        y = row * CELL_HEIGHT
        flat_points.append([x, y, 0.0])

flat_points = np.array(flat_points, dtype=np.float32)


# =====================================================================
#  read the frames and find the corners
# =====================================================================

if not os.path.isdir(FRAMES_DIR):
    print("ERROR: no folder called", FRAMES_DIR)
    sys.exit()

names = os.listdir(FRAMES_DIR)
names.sort()

frames = []
left_list = []
right_list = []

print("reading the frames in", FRAMES_DIR)

for name in names:
    if not name.endswith(".png"):
        continue

    path = os.path.join(FRAMES_DIR, name)
    frame = cv2.imread(path)

    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- find the left wall ---
    found, left = cv2.findChessboardCorners(gray, (LEFT_COLS, LEFT_ROWS), FIND_FLAGS)

    if not found:
        print("  no left wall in", name)
        continue

    left = cv2.cornerSubPix(gray, left, (11, 11), (-1, -1), STOP_RULE)

    # --- find the right wall ---
    found, right = cv2.findChessboardCorners(gray, (RIGHT_COLS, RIGHT_ROWS), FIND_FLAGS)

    if not found:
        print("  no right wall in", name)
        continue

    right = cv2.cornerSubPix(gray, right, (11, 11), (-1, -1), STOP_RULE)

    # --- the left wall must really be on the left ---
    left_x = left[:, 0, 0].mean()
    right_x = right[:, 0, 0].mean()

    if left_x >= right_x:
        print("  walls look swapped in", name)
        continue

    # --- tidy the left wall: top row first, then left to right ---
    grid = left.reshape(LEFT_ROWS, LEFT_COLS, 2)

    top_y = grid[0, :, 1].mean()
    bottom_y = grid[LEFT_ROWS - 1, :, 1].mean()

    if top_y > bottom_y:
        grid = np.flip(grid, axis=0)

    start_x = grid[0, 0, 0]
    end_x = grid[0, LEFT_COLS - 1, 0]

    if start_x > end_x:
        grid = np.flip(grid, axis=1)

    left = grid.reshape(LEFT_ROWS * LEFT_COLS, 1, 2)
    left = left.astype(np.float32)

    # --- tidy the right wall, exactly the same way ---
    grid = right.reshape(RIGHT_ROWS, RIGHT_COLS, 2)

    top_y = grid[0, :, 1].mean()
    bottom_y = grid[RIGHT_ROWS - 1, :, 1].mean()

    if top_y > bottom_y:
        grid = np.flip(grid, axis=0)

    start_x = grid[0, 0, 0]
    end_x = grid[0, RIGHT_COLS - 1, 0]

    if start_x > end_x:
        grid = np.flip(grid, axis=1)

    right = grid.reshape(RIGHT_ROWS * RIGHT_COLS, 1, 2)
    right = right.astype(np.float32)

    frames.append(frame)
    left_list.append(left)
    right_list.append(right)

print("found the board in", len(frames), "frames")

if len(frames) < 5:
    print("not enough frames, run extract_frames.py again with more")
    sys.exit()

height = frames[0].shape[0]
width = frames[0].shape[1]
image_size = (width, height)


# =====================================================================
# 4.1, calibrate with the right wall only
# =====================================================================

object_points = []
image_points = []

for corners in right_list:
    object_points.append(flat_points)
    image_points.append(corners)

result = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

flat_error = result[0]
flat_K = result[1]
flat_dist = result[2]

print("")
print("task 4.1 - right wall only:")
print("  fx =", round(float(flat_K[0, 0]), 2))
print("  fy =", round(float(flat_K[1, 1]), 2))
print("  cx =", round(float(flat_K[0, 2]), 2))
print("  cy =", round(float(flat_K[1, 2]), 2))
print("  error =", round(float(flat_error), 4), "pixels")
print("  distortion =", np.round(flat_dist.ravel(), 5))


# =====================================================================
#  4.2, calibrate with both walls
# =====================================================================

object_points = []
image_points = []

for i in range(len(left_list)):
    both = np.vstack([left_list[i], right_list[i]])
    object_points.append(board_points)
    image_points.append(both)

# a 3D rig needs a starting K, any rough one works
start_K = np.zeros((3, 3))
start_K[0, 0] = float(width)
start_K[1, 1] = float(width)
start_K[0, 2] = width / 2.0
start_K[1, 2] = height / 2.0
start_K[2, 2] = 1.0

start_dist = np.zeros((5, 1))

result = cv2.calibrateCamera(object_points, image_points, image_size,
                             start_K, start_dist,
                             flags=cv2.CALIB_USE_INTRINSIC_GUESS)

full_error = result[0]
full_K = result[1]
full_dist = result[2]
rotations = result[3]
translations = result[4]

print("")
print("task 4.2 - both walls:")
print("  fx =", round(float(full_K[0, 0]), 2))
print("  fy =", round(float(full_K[1, 1]), 2))
print("  cx =", round(float(full_K[0, 2]), 2))
print("  cy =", round(float(full_K[1, 2]), 2))
print("  error =", round(float(full_error), 4), "pixels")
print("  distortion =", np.round(full_dist.ravel(), 5))


# =====================================================================
# 4.3, draw the answer back onto one frame
# =====================================================================

frame = frames[0]
measured = np.vstack([left_list[0], right_list[0]])

projected, _ = cv2.projectPoints(board_points, rotations[0], translations[0],
                                 full_K, full_dist)

total = 0.0
count = len(board_points)

for i in range(count):
    dx = projected[i, 0, 0] - measured[i, 0, 0]
    dy = projected[i, 0, 1] - measured[i, 0, 1]
    total = total + dx * dx + dy * dy

rms = np.sqrt(total / count)

picture = frame.copy()

for i in range(count):
    x = int(round(measured[i, 0, 0]))
    y = int(round(measured[i, 0, 1]))
    cv2.circle(picture, (x, y), 6, (0, 255, 0), 2)

for i in range(count):
    x = int(round(projected[i, 0, 0]))
    y = int(round(projected[i, 0, 1]))
    cv2.drawMarker(picture, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)

cv2.imwrite(CHECK_IMAGE, picture)

print("")
print("task 4.3 - reprojection check on frame 0")
print("  error =", round(float(rms), 4), "pixels")
print("  saved", CHECK_IMAGE, "- green = found, red = predicted")
