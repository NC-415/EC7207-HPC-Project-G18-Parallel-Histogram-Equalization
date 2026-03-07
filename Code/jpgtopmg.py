import cv2
img = cv2.imread("HPCproject.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("input.pgm", img)
