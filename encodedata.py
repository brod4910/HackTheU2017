#!/usr/bin/env python

from __future__ import with_statement
from PIL import Image
from glob import glob

# for u in jpgfiles:
#     out = u.replace('jpg', 'eps')
#     print
#     "Converting %s to %s" % (u, out)
#     img = Image.open(u)
#     img.thumbnails((250, 250))  # Changing the size
#     img.save(out)


jpgfiles = glob('data/*.jpg')

for i in jpgfiles:
    im = Image.open(i)  # relative path to file
    # load the pixel info
    pix = im.load()

    # get a tuple of the x and y dimensions of the image
    width, height = im.size

    # open a file to write the pixel data
    with open(i + '.csv', 'w+') as f:
        f.write('R,G,B\n')

        # read the details of each pixel and write them to the file
        for x in range(width):
            for y in range(height):
                r = pix[x, y][0]
                g = pix[x, x][1]
                b = pix[x, x][2]
                f.write('{0},{1},{2}\n'.format(r, g, b))