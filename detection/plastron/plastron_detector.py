import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='1'
import cv2
import sys
from enum import Enum
from typing import List
import glob
from collections import defaultdict, Counter
from pixellib.instance import custom_segmentation
from shapely.geometry import Polygon, box, Point
import os
from os import path
import math
import json
import argparse
import numpy as np
from skimage import filters, measure
import matplotlib.pyplot as plt
from skimage.draw import rectangle
import copy
from plastron import Utils
from plastron.Utils import drawPolygon, storePolygons


script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
model_path = 'mask_rcnn_model.plastron.h5'
abs_model_path = os.path.join(script_dir, model_path)

segment_image = custom_segmentation()
segment_image.inferConfig(num_classes=4,class_names=["BG", "plastron", "bottom_part", "central_seam", "seam_connection"])
segment_image.load_model(abs_model_path)


def loadLabels(img_path):
    label_path = img_path.replace('jpg', 'json').replace('JPG', 'json')
    ret = defaultdict(lambda: [])
    if not path.exists(label_path):
        return None
    with open(label_path) as json_file:
        data = json.load(json_file)
        for p in data['shapes']:
            # TODO check max x and y
            if p["shape_type"] == "rectangle":
                points = p['points']
                p1 = points[0]
                p2 = points[1]

                minY = int(min(p1[1],p2[1]))
                maxY = int(max(p1[1],p2[1]))
                minX = int(min(p1[0],p2[0]))
                maxX = int(max(p1[0],p2[0]))
                ret[p['label']].append(box(minX,minY,maxX,maxY))
            else:
                tmp = Polygon([(point[0], point[1]) for point in p['points']])
                ret[p['label']].append(box(*tmp.bounds))

    return ret

def mask_to_img(mask):
    mask = np.expand_dims(mask, axis=-1)  # (H, W) -> (H, W, 1)
    return np.concatenate((mask, mask, mask), axis=-1)  # (H, W, 1)

def draw_filled_polygon(img, central_seam:Polygon):
    xs = []
    ys = []
    xx, yy =central_seam.exterior.coords.xy
    xs = xs + [*xx]
    ys = ys + [*yy]

    max_x = max(xs)
    min_x = min(xs)
    max_y = max(ys)
    min_y = min(ys)
    mask = np.zeros(img.shape)
    rr, cc = rectangle((min_y,min_x), end=(max_y,max_x), shape=img.shape)
    rr = rr.astype(np.int)
    cc = cc.astype(np.int)
    mask[rr,cc] = 1
    mask = mask > 0
    return mask

def cut_mask(img, plastron_mask, central_seam:Polygon):
    p_mask = plastron_mask
    if central_seam is not None:
        s_mask = draw_filled_polygon(plastron_mask,central_seam)
        yy,xx = np.nonzero(plastron_mask)
        min_y = min(yy)
        min_y = int(min_y*2)
        s_mask[min_y:, :] = False
        p_mask = plastron_mask | s_mask

    p_mask = mask_to_img(p_mask)
    p_mask = np.invert(p_mask)
    np.putmask(img,p_mask,255)
    return img

def load_masks(pred, class_names):
    classes = np.array(pred['class_ids'])
    masks = pred['masks']
    ret = {}

    for i, cls in enumerate(class_names):
        same_cls = np.array(np.where(classes == i))
        if same_cls.size == 0:
            ret[cls] = None
        else:
            ret[cls] = [masks[:,:,i].reshape(masks[:,:,i].shape[0],-1) for i in same_cls[0]]
    return ret

def loadPolygons(pred, class_names, max_objects=5):
    def toPoligon(roi):
        y1, x1, y2, x2 = roi
        return box(x1, y1, x2, y2)

    ret = {}
    classes = np.array(pred['class_ids'])
    scores = np.array(pred['scores'])
    rois = [toPoligon(r) for r in np.array(pred['rois'])]

    for i, cls in enumerate(class_names):
        same_cls = np.array(np.where(classes == i))
        if same_cls.size == 0:
            ret[cls] = None
        else:
            cls_scores = np.take(scores, same_cls)[0]
            cls_rois = np.take(rois, same_cls)[0]
            n = min(max_objects, cls_scores.shape[0])
            idx = np.argpartition(cls_scores, -n)[-n:]
            max_indexes = idx[np.argsort((-cls_scores)[idx])]

            ret[cls] = np.take(cls_rois, max_indexes, axis=0)

    return ret


def iou_metric(polygon1: Polygon, polygon2: Polygon):
    intersection = polygon1.intersection(polygon2).area,
    union = polygon1.union(polygon2).area
    return intersection[0] / union


def poligonBounds( p: Polygon):
    x, y = p.exterior.xy
    minY = int(min(y))
    maxY = int(max(y))
    minX = int(min(x))
    maxX = int(max(x))
    return minY,minX,maxY,maxX

def cropImage(img, p: Polygon):
    height, width, _ = img.shape
    minY, minX, maxY, maxX  = poligonBounds(p)
    cropped = img[minY:maxY, minX:maxX, :]
    return cropped

def substracFromPolygon(minY, minX, p:Polygon):
    x, y = p.exterior.xy
    x = [xp - minX for xp in x ]
    y = [yp - minY for yp in y ]
    return Polygon(list(map(lambda a, b:(a,b), x, y)))

def scharFilter(img):
    schar = filters.scharr(img)
    info = np.finfo(schar.dtype)  # Get the information of the incoming image type
    data = schar.astype(np.float64) / info.max  # normalize the data to 0 - 1
    data = 255 * data  # Now scale by 255
    img = data.astype(np.uint8)

    equ = cv2.equalizeHist(img)
    inv = (255 - equ)
    return inv


def gaborKernel(img):
    g_kernel = cv2.getGaborKernel((5, 5), 1.0, 180, 1, 1, 0, ktype=cv2.CV_32F)
    return cv2.filter2D(img, cv2.CV_8UC3, g_kernel)


def cannyForCentralSeam(img, sigma=0.8):
    v = np.median(img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img, lower, upper)
    # return the edged image
    edged = (255 - edged)
    return edged


def normalize(im):
    im -= im.min()
    return im / im.max()


def adaptive_threshold(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2);


def floatToInt(float_img):
    info = np.finfo(float_img.dtype)  # Get the information of the incoming image type
    data = float_img.astype(np.float64) / info.max  # normalize the data to 0 - 1
    data = 255 * data  # Now scale by 255
    img = data.astype(np.uint8)
    return img

def get_largest(label):
    label = measure.label(label)
    largest = label == np.argmax(np.bincount(label.flat)[1:]) + 1
    return largest


def get_tight_central_seam_boundbox(central_seam:Polygon, plastron_mask):
    x, y = central_seam.exterior.coords.xy
    ymin = min(y)

    yy, xx = np.nonzero(plastron_mask)

    xmax = max(xx)
    xmin = min(xx)
    ymax = max(yy)


    return box(xmin, ymin, xmax, ymax)


def findBestIou(truth, predicted):
    b_ious = []
    b_poligons = []
    for t in truth:
        best_poligon = None
        best_iou = 0.0
        for p in predicted:
            iou = iou_metric(p, t)
            if iou >= best_iou:
                best_iou = iou
                best_poligon = p
        b_ious.append(best_iou)
        b_poligons.append(best_poligon)
    return b_ious, b_poligons


def measure_ious(result_map, predicted_labels, labels, image):
    if not labels:
        return

    for cls in result_map.keys():

        predicted = predicted_labels[cls]
        truth = None
        if cls in labels:
            truth = labels[cls]

        if predicted is not None and truth is not None:
            b_ious, b_poligons = findBestIou(truth, predicted)

            for bp in b_poligons:
                drawPolygon(image, bp, color=(0, 255, 0))
            for t in truth:
                drawPolygon(image, t, color=(0, 0, 255))
            result_map[cls] = result_map[cls] + b_ious
        else:
            result_map[cls] = result_map[cls] + [0.0]

def getVector(central_seam: Polygon, botom_part: Polygon):
    cs_centroid = central_seam.centroid
    bp_centroid = botom_part.centroid
    return Point(cs_centroid.x - bp_centroid.x, cs_centroid.y - bp_centroid.y)

def length(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


def angle(vector_1, vector_2):
    return -math.degrees(math.asin((vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0])/(length(vector_1)*length(vector_2))))

def rotate_mask(image, angle, center):
    ret = rotate_bound(image.astype(np.uint8), angle, center)
    return ret.astype(np.bool)

def rotate_bound(image, angle, center):

    (h, w) = image.shape[:2]
    (cX, cY) = center

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

def drawPredictedBoxes(_img, predicted_labels, color=(255,0,0)):
    if not predicted_labels:
        return _img

    img = np.copy(_img)
    for v in predicted_labels.values():
        if v is not None:
            for p in v:
                drawPolygon(img, p, color=color)

    if len(predicted_labels['seam_connection']) > 2:
        tc = getToptIntersection(predicted_labels['seam_connection']).centroid
        bc = getLowtIntersection(predicted_labels['seam_connection']).centroid
        cv2.line(img, (int(tc.x), int(tc.y)), (int(bc.x), int(bc.y)), color=(0, 0, 255))

    return img

def rotate_polygon(bb: Polygon, angle, image, cx, cy):
    (h, w) = image.shape[:2]
    x, y = bb.exterior.coords.xy
    old_bb = zip(x, y)
    new_bb = []
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    for i, coord in enumerate(old_bb):
        v = [coord[0], coord[1], 1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M, v)
        new_bb.append((int(calculated[0]), int(calculated[1])))
    return Polygon(new_bb)


def get_two_furthest( p:List[Polygon]):
    max_dist = 0
    ret = None
    for i,p1 in enumerate(p):
        for p2 in p[i+1:]:
            dist = p1.centroid.distance(p2.centroid)
            if dist > max_dist:
                ret = (p1,p2)
                max_dist = dist
    return ret

def filterHashIntersection(intersections:List[Polygon], centralSeam:Polygon)->List[Polygon]:
    return  [p for p in intersections if centralSeam.intersection(p).area != 0]

def filterContains(intersections:List[Polygon], centralSeam:Polygon)->List[Polygon]:
    """
    removes false detections of intersections
    :param intersections:
    :param centralSeam:
    :return:
    """
    return [p for p in intersections if centralSeam.contains(p)]

def filterContainsCentroid(bottomParts:List[Polygon], centralSeam:Polygon)->List[Polygon]:
    """
      removes false detections of bootom part
      :param intersections:
      :param centralSeam:
      :return:
      """
    return [p for p in bottomParts if centralSeam.contains(p.centroid)]

def getToptIntersection(intersections:List[Polygon]):
    i = np.argmax(np.array([p.centroid.y for p in intersections]))
    return intersections[i]

def getLowtIntersection(intersections:List[Polygon]):
    i = np.argmin(np.array([p.centroid.y for p in intersections]))
    return intersections[i]


def findBestBottomPart(bottomParts:List[Polygon],intersections:List[Polygon], centralSeam:Polygon )->Polygon:
    f_intersections = filterHashIntersection(intersections, centralSeam)
    f_bottomParts =  filterContainsCentroid(bottomParts, centralSeam)

    if len(f_intersections) == 5:
        for b_part in f_bottomParts:
            b_intersections =filterHashIntersection(f_intersections, b_part)
            if len(b_intersections) == 2:
                return b_part

    if len(f_bottomParts) > 0:
        return f_bottomParts[0]
    return bottomParts[0]


def rotateAreas(predicted_labels, r_angle, image, center):
    for k in predicted_labels.keys():
        if predicted_labels[k] is not None:
            predicted_labels[k] = [rotate_polygon(p, r_angle, image, center.x, center.y) for p in predicted_labels[k]]

def labels_to_str(predicted_labels):
    return " cs:" + str(col_len(predicted_labels['central_seam'])) + \
           " bp:" + str(col_len(predicted_labels['bottom_part'])) + \
           " i:" + str(col_len(predicted_labels['seam_connection'])) + \
           " p:" + str(col_len(predicted_labels['plastron']))

def col_len(col):
    if col is None:
        return None
    return len(col)


def predict(image):
    pred, output = segment_image.segmentFrame(image, show_bboxes=True)
    return pred, output

def merge_plastron_and_bottom_mask(plastron_mask, bottom_mask):
    yy, xx = np.nonzero(bottom_mask)
    max_y = max(yy)
    min_y = min(yy)
    h = int((max_y - min_y)/5)
    ret = np.copy(plastron_mask)
    ret[min_y+h:, :] = False
    ret = ret | bottom_mask
    return ret

def detect_plastron(image, image_id=None, test_labels=None):
    class_names = segment_image.config.class_names

    pred, output = predict(np.copy(image))

    predicted_masks = load_masks(pred, class_names)
    predicted_labels = loadPolygons(pred, class_names)
    original_predicted_labels = copy.deepcopy(predicted_labels)

    if predicted_masks['plastron'] is None:
        raise Exception('detection failed')

    plastron_mask = predicted_masks['plastron'][0]
    if predicted_labels['central_seam'] is not None and predicted_labels['bottom_part'] is not None:
        bottom_mask = predicted_masks['bottom_part'][0]
        seam_center = predicted_labels['central_seam'][0]
        bottom_parts = predicted_labels['bottom_part']

        predicted_labels['seam_connection'] = filterContainsCentroid(predicted_labels['seam_connection'], seam_center)
        intersections = predicted_labels['seam_connection']

        best_bottom_part = findBestBottomPart(bottom_parts, intersections, seam_center)
        predicted_labels['bottom_part'] = [best_bottom_part]

        vec = getVector(seam_center, best_bottom_part)
        cs_centroid = seam_center.centroid

        r_angle = angle([0, 1], [vec.x, vec.y])
        r_image = rotate_bound(image, r_angle, center=(cs_centroid.x, cs_centroid.y))
        plastron_mask = rotate_mask(plastron_mask, r_angle, center=(cs_centroid.x, cs_centroid.y))
        bottom_mask = rotate_mask(bottom_mask, r_angle, center=(cs_centroid.x, cs_centroid.y))
        rotateAreas(predicted_labels, r_angle, image, cs_centroid)
        if test_labels:
            rotateAreas(test_labels, r_angle, image, cs_centroid)


        if len(intersections) > 1:
            top = getToptIntersection(predicted_labels['seam_connection'])
            bot = getLowtIntersection(predicted_labels['seam_connection'])

            i_vec = getVector(bot, top)
            i_angle = angle([0, 1], [i_vec.x, i_vec.y])

            cs_centroid = predicted_labels['central_seam'][0].centroid

            final_image = rotate_bound(r_image, i_angle, center=(cs_centroid.x, cs_centroid.y))
            plastron_mask = rotate_mask(plastron_mask, i_angle, center=(cs_centroid.x, cs_centroid.y))
            bottom_mask = rotate_mask(bottom_mask, i_angle, center=(cs_centroid.x, cs_centroid.y))
            rotateAreas(predicted_labels, i_angle, r_image, cs_centroid)
            if test_labels:
                rotateAreas(test_labels, i_angle, r_image, cs_centroid)

            r_image = final_image
    else:
        print(str(image_id) + "  FUCK!" + labels_to_str(predicted_labels))
        raise Exception('detection failed')

    try:
        plastron_mask = merge_plastron_and_bottom_mask(plastron_mask,bottom_mask)

        tight_central_seam = get_tight_central_seam_boundbox(
            predicted_labels['central_seam'][0],
            plastron_mask
        )

        r_image = cut_mask(r_image, plastron_mask,  predicted_labels['central_seam'][0])

        predicted_labels['crop'] = [tight_central_seam]

        drawn_img = drawPredictedBoxes(r_image, predicted_labels)
        cropped = cropImage(r_image, tight_central_seam)

        minY, minX, _, _ = poligonBounds(tight_central_seam)
        croped_intersections = [substracFromPolygon(minY, minX, p) for p in predicted_labels['seam_connection']]

        if test_labels:
            test_labels = {
                 k : [substracFromPolygon(minY, minX, p) for p in v] for k,v in test_labels.items()
            }
        # croped_junctions = [substracFromPolygon(minY, minX, p) for p in labels['junction']]

        # for p in croped_junctions:
        # Utils.drawPolygon(copy_cropped, p)
        # storePolygons(img_path_junctions_curated, croped_junctions)

    except Exception as e:
        print(e)
        raise Exception('detection failed')
    print(str(image_id) + "  GOOD! " + labels_to_str(predicted_labels))
    return cropped, croped_intersections, drawn_img, original_predicted_labels, test_labels


def checkDetections(labels,original_predicted_labels, rotated_labels ):
    if original_predicted_labels['central_seam'] is None:
        return 'missing_central_seam'


    if original_predicted_labels['bottom_part'] is None \
            or len(original_predicted_labels['bottom_part']) < 1:
        return 'missing_seam_bottom_part'


    if labels:
        for cls in ['central_seam','bottom_part']:
            if cls in labels:
                b_ious, b_poligons = findBestIou(labels[cls], [original_predicted_labels[cls][0]])
                min_iou = min(b_ious)
                if min_iou < 0.1:
                    return'bad_location_' + str(cls)

        if 'seam_connection' in labels:
            if len(original_predicted_labels['seam_connection']) < 5:
                return 'missing_seam_connection'

            b_ious, b_poligons = findBestIou(labels['seam_connection'], original_predicted_labels['seam_connection'])
            min_iou = min(b_ious)
            if min_iou < 0.01:
                return 'bad_location_' + str(cls)
        else:
            return 'missing_seam_connection'

    if rotated_labels:
        bot = rotated_labels['bot'][0]
        top = rotated_labels['top'][0]
        if bot.centroid.y < top.centroid.y:
            return 'orientation'

    return 'correct'


def run_test_detection(path, output):
    files = glob.glob(path + os.path.sep + '*.jpg')
    count = len(files)
    print('runing detection test on:' + str(count) + ' files.')
    print("\n\n\n")
    result_map = {
        'plastron': [],
        'bottom_part': [],
        'central_seam': [],
        'seam_connection': []
    }
    counter = Counter()
    faulty_ids = defaultdict(list)
    for img_path in files:
        try:
            image = cv2.imread(img_path)
            tg_id = os.path.splitext(os.path.basename(img_path))[0]
            labels = loadLabels(img_path)
            labels_copy = None
            if labels:
                labels_copy = copy.deepcopy(labels)

            cropped, croped_intersections, marked_image, original_predicted_labels,test_labels = detect_plastron(image, image_id=tg_id, test_labels=labels_copy)

            draw = np.copy(cropped)
            marked_original = drawPredictedBoxes(image, original_predicted_labels, color=(255,255,255))

            compare_image = drawPredictedBoxes(image, original_predicted_labels, color=(255,255,255))

            measure_ious(result_map,original_predicted_labels,labels, image)
            result = checkDetections(labels, original_predicted_labels, labels_copy)
            faulty_ids[result].append(tg_id)
            counter[result]+=1

            if result == 'correct':
                polys = [p for p in test_labels['junction']] + [p for p in test_labels['bot']] + [p for p in test_labels['top']]
                Utils.storePolygons(output+'/test_labels_'+str(tg_id)+'.json', polys)

                cv2.imwrite(output+'/'+str(tg_id)+'.png', marked_image)

                json_file_path = output + '/cropped_' + str(tg_id) + '.json'
                storePolygons(json_file_path, croped_intersections)

                cropped_image_file_path = output + '/cropped_' + str(tg_id) + '.png'
                cv2.imwrite(cropped_image_file_path, cropped)
            else:
                print("incorrect")
                cv2.imwrite(output + '/aa_incorect_' + str(tg_id) + '.png', marked_original)
        except Exception as e:
            counter['exception']+=1
            print(e)
            continue

    f_label_map = {
        'plastron': 'IOU histogram třídy Plastron',
        'bottom_part': 'IOU histogram třídy Spodní část',
        'central_seam': 'IOU histogram třídy Centrální šev',
        'seam_connection': 'IOU histogram třídy Spoj'
    }

    print(json.dumps(result_map))

    for k in result_map.keys():
        fig, ax = plt.subplots()
        ax.hist(result_map[k], bins=30)
        ax.set_ylabel('Počet')
        ax.set_xlabel('IOU')
        ax.set_title(f_label_map[k])

        fig.savefig('figs/new_iou_hist_' + k + '.png', bbox_inches='tight')
        plt.clf()

    print('FAULTY IDS:')
    print(faulty_ids)

    w_label_map = {
        'missing_plastron': 'Nenalezen Plastron',
        'missing_central_seam': 'Nenalezen Centrální šev',
        'missing_seam_bottom_part': 'Nenalezena Spodní část',
        'missing_seam_connection': 'Nenalezen spoj',
        'bad_location_central_seam': 'Poloha centralního švu',
        'bad_location_bottom_part' : 'Poloha spodní části',
        'bad_location_seam_connection'  :  'Poloha spoje',
        'orientation': 'Orientace plastronu',
        'correct': 'Úspěšná detekce',
        'exception': 'Chyba aplikace',
    }


    values = [v/count for v in  counter.values()]
    fig, ax = plt.subplots()
    y_pos = np.arange(len(counter.keys()))
    plt.barh(y_pos , values)

    for i, v in enumerate(values):
        ax.text(max(v - 0.2, 0.15), i, "{:.2f}".format(v), color='black', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([ w_label_map[k] for k in counter.keys()])
    ax.set_xlabel('Počet')
    ax.set_title('Vyhodnocení úspěšnosti detekce.')
    fig.savefig('figs/error.png', bbox_inches='tight')


def run_detection_and_preprocessesing(path, out_folder='c:/outputs/marked_2/'):
    files = glob.glob(path+ os.path.sep +'*.jpg')
    print("running detetion for: " + str(len(files)) + ' files')
    print("\n\n\n")
    for img_path in files:
        image = cv2.imread(img_path)
        tg_id = os.path.splitext(os.path.basename(img_path))[0]
        try:
            cropped, croped_intersections, marked_image, original_predicted_labels,_ = detect_plastron(image, image_id=tg_id)

            json_file_path = out_folder + 'cropped_' + str(tg_id) + '.json'
            storePolygons(json_file_path, croped_intersections)

            cropped_image_file_path = out_folder + 'cropped_' + str(tg_id) + '.png'
            cv2.imwrite(cropped_image_file_path, cropped)
        except Exception as e:
            continue

        marked_image_file_path = out_folder + 'marked_' + str(tg_id) + '.png'
        #cv2.imwrite(marked_image_file_path, marked_image)


class MODE(Enum):
    DETECT=1
    TEST=2

    @staticmethod
    def from_string(s):
        try:
            return MODE[s]
        except KeyError:
            raise ValueError()


parser = argparse.ArgumentParser()
parser.add_argument('-m','--mode', help="labels are necessary for Test.", choices=[m.name for m in MODE])
parser.add_argument('-i','--input', help="input folder with plastron images.")
parser.add_argument('-o','--output', help="input folder with plastron images.")
parser.parse_args()


args = parser.parse_args()

if __name__ == "__main__":
    parsed_args = parser.parse_args(sys.argv[1:])
    try:
        mode = parsed_args.mode
        input_path = parsed_args.input
        output_path = parsed_args.output
    except Exception as e:
        print(e)
        parser.print_help()
        exit(-1)

    if mode == 'DETECT':
        run_detection_and_preprocessesing(input_path, out_folder=output_path)
    elif mode == 'TEST':
        run_test_detection(input_path, output_path)