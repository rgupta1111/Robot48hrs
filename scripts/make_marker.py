"""Write the ArUco marker for hand-eye calibration: data/aruco_marker_id0.png (4x4_50, id 0), white border included.

Print it so the BLACK SQUARE is a size you can measure (80 mm by default), keep it flat and matte (tape it on the table), and tell
src/calibrate_handeye.py the measured size with --marker-mm.
"""
import os

import cv2

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
img = cv2.aruco.generateImageMarker(d, 0, 600)                       # 600 px black square
img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)   # white quiet zone
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "aruco_marker_id0.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
cv2.imwrite(out, img)
print("wrote", os.path.abspath(out), "- print it so the black square measures a known size (default 80 mm)")
