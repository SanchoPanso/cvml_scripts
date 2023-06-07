import cv2
import numpy as np
import colorsys


def get_palette(num_of_colors: int) -> list:
    hsv_tuples = [(x * 1.0/ num_of_colors, 0.5, 0.5) for x in range(num_of_colors)]
    rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
    rgb_tuples = list(map(lambda x: (int(x[1] * 255), int(x[0] * 255), int(x[2] * 255)), rgb_tuples))
    return rgb_tuples

num_of_colors = 15
palette = get_palette(num_of_colors)
print(palette)

img = np.array(palette, dtype='int8').reshape(1, num_of_colors, 3)
cv2.imshow("1", img)
cv2.waitKey()