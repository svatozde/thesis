import numpy as np
import math

def test(img, weights):
    height = img.shape[0]
    width = img.shape[1]
    m_i = math.floor( weights.shape[0]/2)
    m_j = weights.shape[1]

    for h,w in np.transpose(np.nonzero(img)):
        ymin = max(0 , h-m_i)
        ymax = min(height, h+m_i+1)
        xmin = w
        xmax = min(width, w + m_j)
        masked = img[ymin:ymax, xmin:xmax]
        print(str(h)+ ':' + str(w))
        #print(masked)
        print()
        print()
        for y, x in np.transpose(np.nonzero(masked)):
            th = ymin + y
            tw = xmin + x
            wy = ymin - h + m_i + y
            weight = weights[wy][x]
            print(str(img[h][w])+'->'+str(img[th][tw]) + '=' +str(weight))


mask = np.array(
    [
        [10, 20, 30,40,50,60],
    ]
).astype(np.double)

weights = np.array(
    [
        [11, 12, 13, ],
        [21, 22, 23, ],
        [31, 32, 33, ],
        [41, 42, 43, ],
        [51, 52, 53, ],
    ]
)

test(mask, weights)
