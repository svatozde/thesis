import argparse
import json
import os
import sys
import time

from matplotlib.patches import Polygon
from skimage.color import rgb2gray
from skimage.draw import disk
from skimage.morphology import thin, skeletonize
from skimage.transform import rotate
from skimage.feature import canny
from skimage.filters import meijering, try_all_threshold, threshold_otsu, threshold_mean
from skimage.feature.corner import corner_harris
from skimage.io import imsave, imread

import GADS
import cv2
import numpy as np
import glob
import warnings

from plastron import Utils
from shapely.affinity import translate

warnings.filterwarnings("ignore")


def get_density(img):
    return np.count_nonzero(img) / img.size


def get_nonzero_limits(canny_img):
    yy, xx =np.nonzero(canny_img)
    max_r = max(yy)
    min_r = min(yy)
    return min_r,max_r



def process(pth , out):
    tg_id = os.path.splitext(os.path.basename(pth))[0]

    original = imread(pth)
    gray = rgb2gray(original)
    rgb_o = gray
    c_img = canny(
        gray,
        sigma=1.0,
        use_quantiles=True,
        low_threshold=0.0,
        high_threshold=0.0
    )

    mimg = meijering(rgb_o, sigmas=range(1, 3, 1), black_ridges=True)
    mimg = mimg > threshold_mean(mimg)
    mimg = thin(mimg, max_iter=4)
    img = c_img + mimg

    plastron_edge = np.zeros(c_img.shape)
    for hh in range(0, img.shape[0]):
        if (hh % 10 == 0):
            row = np.array(np.nonzero(img[hh, 5:-5]))
            if row.size > 0:
                plastron_edge[hh][:np.min(row)+10] = 1
                plastron_edge[hh][np.max(row)-10:] = 1

    #Silouhete of plasstron
    img = img + plastron_edge

    min_y, max_y = get_nonzero_limits(img)
    img = img[min_y:max_y, :]
    rgb_o = rgb_o[min_y:max_y, :]

    polygons = Utils.loadPolygons(pth.replace('.png', '.json'))
    polygons = [translate(p, yoff=-min_y) for p in polygons]
    if len(polygons) < 5:
        print('wrong-masks!')
        return
    points_json = {
        'left': [],
        'right': []
    }

    masks = [
        Utils.create_upper_mask(img, polygons),
        *Utils.partial_masks(img, polygons, multiply_height=1.5),
        Utils.create_lower_mask(img, polygons)
    ]

    central_masks = [
        Utils.create_upper_mask(img, polygons),
        *Utils.partial_masks(img, polygons, multiply_height=1.0),
        Utils.create_lower_mask(img, polygons)
    ]

    _, _, middle_mask = Utils.cs_mask(img, polygons, multiply_height=-0.7)
    tmp_img = img * middle_mask

    tmp_img = rotate(tmp_img, 90, resize=True)
    rgb_o = rotate(rgb_o, 90, resize=True)
    (height, width) = img.shape
    tmp_img[:, 0:20] = 1
    tmp_img[:, -20:] = 1

    node_count = np.count_nonzero(img)
    print(pth + ' pcount: ' + str(node_count) + ' shape: ' + str(img.shape) + ' m: ' + str(len(polygons)))

    result_seam = np.zeros(tmp_img.shape).astype(np.bool)

    weights = GADS.s_dist_mat(7, GADS.dist_matej)
    builder = GADS.binaryArrayIntoOrientedRedundantGraph
    weights_2 = GADS.rect_dist_mat(7, GADS.dist_piramid)
    builder_2 = GADS.binaryArrayIntoOrientedRedundantGraph
    start = time.time()
    # print(substract)

    substract = 5
    min_path = ret_cs = None
    while min_path is None:
        print('substract: ' + str(substract))
        min_path, _, ret_cs, _, _ = GADS.gads(
            tmp_img,
            graph_builder=builder,
            weights=weights,
            end_width=substract,
            start_width=substract
        )
        substract += 50

    if min_path is None:
        print('failed to find central seam!')
        return
    end = time.time()

    print('Central time: ' + str(end - start))

    np.putmask(result_seam, ret_cs, 1)
    np.putmask(rgb_o, ret_cs, 1)
    rgb_o = rotate(rgb_o, -90, resize=True)
    result_seam = rotate(result_seam, -90, resize=True)
    central_seam_only = np.copy(result_seam)

    left_junctions = []
    right_junctions = []

    detected_paths = []
    for i, m_t in enumerate(masks):

        m_min, m_max, m = m_t
        cmin, cmax, cm = central_masks[i]
        tmp_img = img[m_min:m_max, :]
        tmp_edge = plastron_edge[m_min:m_max, :]
        tmp_res = (cm * central_seam_only)[m_min:m_max, :]
        tmp_img[:, 0] = 1
        tmp_img[:, -1] = 1
        tmp_img = tmp_img.astype(np.double)
        tmp_o = rgb_o[m_min:m_max, :]
        harris = corner_harris(tmp_o)
        np.putmask(tmp_o, tmp_res, 0)
        np.putmask(tmp_img, tmp_res, 1)
        #tmp_res = harris*tmp_res
        tmp_res = tmp_res > 0

        print('detecting: ' + str(i) + ' size: ' + str(tmp_img.shape))

        start = time.time()
        # flipping
        tmp_res_1 = np.flip(tmp_res, 1)
        seam_points = Utils.get_non_zero_points(tmp_res_1)
        tmp_img_mir = np.flip(tmp_img, 1)

        min_path, _, ret_left, _, _  = GADS.gads(
            tmp_img_mir,
            graph_builder=builder_2,
            start_nodes=seam_points,
            weights=weights_2,
        )
        end = time.time()
        # flip result back
        ret_left = np.flip(ret_left, 1)

        print('detected: ' + str(i) + ' in: ' + str(end - start))
        if min_path is None:
            print('failed to find left side seam seam:' + str(i) + '!')
            imsave(
                output_path + '/' + str(tg_id) + '_mask_left_fail_' + str(i) + '.png',
                (tmp_img * 255).astype(np.uint8)
            )
            continue

        intersection = ret_left * tmp_res
        nonzero_seam = Utils.get_non_zero_points(tmp_res)
        for p in Utils.get_non_zero_points(intersection):
            if p in nonzero_seam:
                p = (p[0] + m_min, p[1])
                left_junctions.append(p)
                break;

        replace_left = np.zeros(rgb_o.shape)
        replace_left[m_min:m_max, :] = ret_left

        np.putmask(rgb_o, replace_left, 1)
        np.putmask(result_seam, replace_left, 1)

        start = time.time()
        min_path, _, ret_right, _, _  = GADS.gads(
            tmp_img,
            graph_builder=builder_2,
            start_nodes=Utils.get_non_zero_points(tmp_res),
            weights=weights_2,
        )
        end = time.time()
        print('detected: ' + str(i) + ' in: ' + str(end - start))

        if min_path is None:
            print('failed to right side seam seam:' + str(i) + '!')
            imsave(
                output_path + '/' + str(tg_id) + '_mask_right_fail_' + str(i) + '.png',
                (tmp_img * 255).astype(np.uint8)
            )
            continue

        intersection = ret_right * tmp_res
        nonzero_seam = Utils.get_non_zero_points(tmp_res)
        for p in Utils.get_non_zero_points(intersection):
            if p in nonzero_seam:
                p = (p[0] + m_min, p[1])
                right_junctions.append(p)
                break

        replace_right = np.zeros(rgb_o.shape)
        replace_right[m_min:m_max, :] = ret_right

        np.putmask(rgb_o, replace_right, 1)
        np.putmask(result_seam, replace_right, 1)



    for j in left_junctions:
        points_json['left'].append((int(j[0]), int(j[1])))
        rr, cc = disk(j, 4, shape=rgb_o.shape)
        rgb_o[rr, cc] = 1

    for j in right_junctions:
        points_json['right'].append((int(j[0]), int(j[1])))
        rr, cc = disk(j, 4, shape=rgb_o.shape)
        rgb_o[rr, cc] = 1

    json.dump(
        points_json,
        open(
            output_path + 'detected_' + str(tg_id) + '.json',
            mode='w'
        )
    )
    imsave( output_path + 'detected_' + str(tg_id) + '.png', rgb_o)
    imsave(output_path + 'binary_' + str(tg_id) + '.png', img)

def run_detection(path, out):
    pths = glob.glob(path+'cropped_Tg*a.png') + glob.glob(path+'cropped_Tg*b.png')
    pths.sort()
    pths = pths + glob.glob(path+'/cropped_Tg*-.png')
    print("running detection for: " + str(len(pths)) + " plastrons.")
    for pth in pths:
        try:
            process(pth, out)
        except Exception as e:
            print('exception' + str(e))


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help="input folder with plastron images.")
parser.add_argument('-o', '--output', help="output folder with detected plastron images and coordinates in json.")
parser.parse_args()
args = parser.parse_args()

if __name__ == "__main__":
    try:
        parsed_args = parser.parse_args(sys.argv[1:])
        input_path = parsed_args.input
        output_path = parsed_args.output
        run_detection(input_path, output_path)
    except Exception as e:
        print(e)
        parser.print_help()
        exit(-2)


