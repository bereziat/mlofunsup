import numpy as np
import os
from math import *
import matplotlib.pyplot as plt


# WARNING: this will work on little-endian architectures (eg Intel x86) only!
f = open("04574_flow.flo",'rb')

m = np.fromfile(f,np.float32,count =1)



w = np.fromfile(f, np.int32, count=1)
h = np.fromfile(f, np.int32, count=1)
print 'Reading %d x %d flo file' % (w, h)
data = np.fromfile(f, np.float32, count=2*w*h)
# Reshape data into 3D array (columns, rows, bands)
#print np.size(data)

data2D = np.resize(data, (h[0], w[0],2))
#plt.imshow(data2D[:,:,1])
#plt.show()
print np.size(data2D,0)
def makeColorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))    #r g b
    col = 0

    ### RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = (np.floor(255 * np.arange(RY) / RY)).transpose()
    col = col + RY

    ### YG
    colorwheel[col:col+YG, 0] = (255 - np.floor(255 * np.arange(YG) / YG)).transpose()
    colorwheel[col:col+YG, 1] = 255
    col = col + YG

    ### GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = (np.floor(255 * np.arange(GC) / GC)).transpose()
    col = col + GC

    ### CB
    colorwheel[col:col+CB, 1] = (255 - np.floor(255 * np.arange(CB) / CB)).transpose()
    colorwheel[col:col+CB, 2] = 255
    col = col + CB

    ### BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = (np.floor(255 * np.arange(BM) / BM)).transpose()
    col = col + GC

    ### MR
    colorwheel[col:col+MR, 2] = (255 - np.floor(255 * np.arange(MR) / MR)).transpose()
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def computeColor(u,v):



    img = np.zeros((np.size(u,0),np.size(u,1),3))
    a = np.zeros(np.shape(u))
    rad = np.sqrt(u**2 + v**2)
    for i in range(np.size(u,0)):
        for j in range(np.size(u,1)):
            a[i,j] = atan2(-v[i,j], -u[i,j]) / pi
    colorwheel = makeColorwheel()
    ncols = np.size(colorwheel,0)
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk)
    #print k0
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0
    col0 = k0 * 0
    col1 = k1 * 0
    for i in range(np.size(colorwheel,1)):

        tmp = colorwheel[:, i]
        for i0 in range(np.size(k0,0)):
            for j0 in range(np.size(k0,1)):
                col0[i0,j0] = tmp[int(k0[i0,j0])-1] / 255
                col1[i0,j0] = tmp[int(k1[i0,j0])-1] / 255
        col = (1 - f)* col0 + f* col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx]* (1 - col[idx])  # increase saturation with radius
        col[~idx] = col[~idx]*0.75
        img[:,:, i] = np.uint8(np.floor(255 * col))
    return img

u = data2D[:,:,0]

v = data2D[:,:,1]

img = computeColor(u,v)
#print img

plt.imshow(img)
plt.show()





