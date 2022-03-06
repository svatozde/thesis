import json
from typing import List, Dict

from shapely.geometry import Polygon, mapping, MultiPolygon, shape, Point
import numpy as np
import cv2
from skimage import draw


def drawPolygons(image, polygons: List[Polygon], color=(255, 255, 0)):
    for p in polygons:
        if p is not None:
            drawPolygon(image, p, color)

def drawPolygon(image, p: Polygon, color=(255, 255, 0)):
    if p is None:
        return
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = [int_coords(p.exterior.coords)]
    cv2.polylines(image, exterior, True, color,thickness=4)


def dumpPolygonMap(polygons:Dict[str,Polygon]):
    ret = {k: dumpPolygons(w) for k,w in polygons.items() }
    return json.dumps(ret)

def dumpPolygons(polygons:List[Polygon]):
    if len(polygons) == 1:
        return json.dumps([mapping(polygons[0])])
    return json.dumps(mapping(MultiPolygon(polygons)))

def storePolygons(json_file_path,polygons:List[Polygon]):
    with open(json_file_path, 'w', encoding='utf-8') as f:
        f.write(dumpPolygons(polygons))

def loadPolygons(json_file_path)->List[Polygon]:
   return sorted(list(shape(json.load(open(json_file_path)))), key=lambda a: a.centroid.y, reverse=True)


def mask(_img, polygons:List[Polygon]):
    mask = np.zeros(_img.shape)
    for p in polygons:
        mask = np.add(mask, mmask(_img, p))
    return mask

def partial_masks(_img, polygons:List[Polygon],multiply_height=0):
    polygons = sorted(polygons, key=lambda p: p.centroid.y)
    return [mmask(_img, p, multiply_height=multiply_height ) for p in polygons]

def mmask(_img, p: Polygon,multiply_height=2):
    mask = np.zeros(_img.shape)
    x, y = p.exterior.xy


    rr, _ = draw.polygon(y, x)
    max_r = max(rr)
    min_r = min(rr)

    h = int((multiply_height*(max_r - min_r))/2)

    max_r = min(_img.shape[0], max_r + h)
    min_r = max(0, min_r - h)

    mask[min_r:max_r, :] = 1
    return  min_r, max_r, mask

def get_non_zero_points(img):
    x,y = np.nonzero(img)
    return [ a for a in zip(x, y)]

def cs_mask(_img, p:List[Polygon],multiply_height=3):
    mask = np.zeros(_img.shape)
    xs = []
    for pp in p:
        x, _ = pp.exterior.xy
        xs += x

    maxX = min(_img.shape[1] ,int(max(xs)))
    minX = max(0, int(min(xs)))

    w = int((multiply_height * (maxX - minX)) / 2)

    maxX = maxX + w
    minX = minX - w

    mask[:, minX:maxX] = 1
    return minX,maxX,  mask


def create_upper_mask(_img, p:List[Polygon]):
    mask = np.zeros(_img.shape)
    ys = []
    for pp in p:
        x, y = pp.exterior.xy
        ys += y
    ymin = int(min(ys))
    sub = int((ymin) / 3)
    mask[sub:ymin, :] = 1
    return 0,ymin,mask


def create_lower_mask(_img,  p:List[Polygon]):
    mask = np.zeros(_img.shape)
    ys = []
    for pp in p:
        x, y = pp.exterior.xy
        ys+=y
    ymax = int(max(ys))
    mask[ymax:_img.shape[0]-40, :] = 1
    return ymax,_img.shape[0],mask

def getClosestPoint(p:Point, points:List[Point]):
    min_dist = 99999999
    ret = None
    for pp in points:
        t_dist = pp.distance(p)
        if t_dist < min_dist:
            ret = pp
            min_dist = t_dist
    return ret
