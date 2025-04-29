from PIL import Image
import math
import numpy as np
img = Image.open('./mission1.png')
img2 = img.resize((269,324))
img2.save('resized.png')
width, height = img2.size
pix_values = list(img2.getdata())
def get_distance(pix1,pix2):
    sqr_dist = (pix1[0] - pix2[0])**2 + (pix1[1] - pix2[1])**2 + (pix1[2] - pix2[2])**2
    return math.sqrt(sqr_dist)
'''
Original Image(mission1.png) -> 13.39cm x 11,09cm
5cm = 12.1km 1cm = 2.4km
13.39cm =32.4km =grid324개
11.09cm = 26.8km=grid269개


[157,199,232] blue
[132,138,148] land
[216,216,214] black
'''
occ_grid = [[0 for i in range(269)] for j in range(324)]
for i in range(269):
    for j in range(324):
        pix = pix_values[width*j+i]
        blue = [157,199,232]
        dist = get_distance(blue,pix)
        if dist<20:
            occ_grid[j][i] = 1
np.savetxt('occupacy_map.dat',np.asarray(occ_grid))
a = np.loadtxt('occupacy_map.dat')