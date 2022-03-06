import glob
import json
import os
import random
from typing import List
import numpy as np
from shapely.geometry import Point
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from scipy import stats
from statistics import mode

from plastron import Utils


class PointPair:
    def __init__(self, a:Point, b:Point):
        if a.x < b.x:
            self.left = a
            self.right = b
            self.dist = a.distance(b)
        else:
            self.left = b
            self.right = a
            self.dist = a.distance(b)

        if self.left.y > self.right.y:
            self.right_higher = True
            self.d = a.distance(b)
        else:
            self.right_higher = False
            self.d = - a.distance(b)

    def get_right_distnce(self,o):
        #works only in direction top down
        if o.right_higher:
            if self.right_higher:
                return self.right.distance(o.left) + o.dist
            else:
                return self.left.distance(o.left) + o.dist + self.dist
        else:
            if self.right_higher:
                return self.right.distance(o.right)
            else:
                return self.left.distance(o.right) + self.dist

    def get_left_distnce(self, o):
        # works only in direction top down
        if o.right_higher:
            if self.right_higher:
                return self.right.distance(o.left) + self.dist
            else:
                return self.left.distance(o.left)
        else:
            if self.right_higher:
                return self.right.distance(o.right) + self.dist + o.dist
            else:
                return self.left.distance(o.right) + o.dist


    def top_right_distnce(self, p:Point):
        if self.right_higher:
            return self.right.distance(p)
        return self.left.distance(p) + self.dist

    def top_left_distnce(self, p: Point):
        if self.right_higher:
            return self.right.distance(p) + self.dist
        return self.left.distance(p)

    def bot_left_distnce(self, p: Point):
        if self.right_higher:
            return self.left.distance(p)
        return self.right.distance(p) + self.dist

    def bot_right_distnce(self, p: Point):
        if self.right_higher:
            return self.left.distance(p) + self.dist
        return self.right.distance(p)


def createPairs(points):
    top = points[0]
    bot = points[-1]

    points = points[1:-1]


    ret = [
        PointPair(points[0],points[1]),
        PointPair(points[2], points[3]),
        PointPair(points[4], points[5]),
        PointPair(points[6], points[7]),
        PointPair(points[8], points[9]),
    ]
    return top,bot,ret


def to_euclideanfeature_vector(top:Point, bot:Point, points:List[PointPair]):
    r2 = points[1].right.distance(points[0].right)
    l2 = points[1].left.distance(points[0].left)

    r3 = points[2].right.distance(points[1].right)
    l3 = points[2].left.distance(points[1].left)
    r4 = points[3].right.distance(points[2].right)
    l4 = points[3].left.distance(points[2].left)
    r5 = points[4].right.distance(points[3].right)
    l5 = points[4].left.distance(points[3].left)

    ret = np.array([l2, l3, l4, l5, r2, r3, r4, r5,])
    return ret

def to_feature_vector(top:Point, bot:Point, points:List[PointPair]):
    r1 = points[0].top_right_distnce(top)
    l1 = points[0].top_left_distnce(top)
    r2 = points[1].get_right_distnce(points[0])
    l2 = points[1].get_left_distnce(points[0])

    r3 = points[2].get_right_distnce(points[1])
    l3 = points[2].get_left_distnce(points[1])
    r4 = points[3].get_right_distnce(points[2])
    l4 = points[3].get_left_distnce(points[2])
    r5 = points[4].get_right_distnce(points[3])
    l5 = points[4].get_left_distnce(points[3])
    r6 = points[4].bot_right_distnce(bot)
    l6 = points[4].bot_left_distnce(bot)

    d1 = points[0].d
    d2 = points[1].d
    d3 = points[2].d
    d4 = points[3].d
    d5 = points[4].d

    #TODO correct sem length
    sum_l = sum([l1,l2, l3, l4, l5, l6,])
    sum_r = sum([r1, r2, r3, r4, r5, r6,])
    #sum_l = sum([l2, l3, l4, l5,])
    #sum_r = sum([r2, r3, r4, r5,])
    assert int(sum_l) == int(sum_r)
    ret = np.array([l1,l2, l3, l4, l5, l6, r1, r2, r3, r4, r5, r6, d1, d2, d3, d4, d5])
    #ret = np.array([l2, l3, l4, l5, r2, r3, r4, r5,])
    ret = ret/sum_r
    return ret


def filter_only_pairs(y):
    vals, counts = np.unique(y, return_counts=True)
    indexes = np.argwhere(counts > 1)
    fvals = vals[indexes];
    ix = np.where(np.isin(y, fvals))
    ret = y[ix]
    return ret

def countHits(Y,y,y_pred_n):
    nth = []
    for i,truth in enumerate(y):
        pp = y_pred_n[i]
        append = -1
        labels = [Y[p] for p in pp]
        for i, pred in enumerate(labels[1:]): #ignore first since it is in model
            if pred == truth:
                append = i+1
                break
        nth.append(append)
    return np.array(nth)



c_n = 20
c_map = plt.cm.get_cmap('hsv', c_n)

def random_color():
    return c_map(random.randint(0,c_n) )

shapes = np.array(['.','<','>','^','v','8','+','x','*',])

def get_shape(cnt):
    return shapes[cnt%shapes.shape[0]]


def load_hand_measured_fetures(pths):
    X = []
    Y = []
    result_points = []
    for pth in pths:
        try:
            tg_id = os.path.splitext(os.path.basename(pth))[0][-len('Tg001-'):]
            print(tg_id)
            polygons = Utils.loadPolygons(pth)
            points = [p.centroid for p in polygons]
            points = sorted(points, key=lambda p: p.y)
            top, bot, pairs = createPairs(points)
            result_points.append((top, bot, pairs))
            feature_vec = to_feature_vector(top, bot, pairs)
            #feature_vec = to_euclideanfeature_vector(top, bot, pairs)
            Y.append(tg_id)
            X.append(feature_vec)
        except Exception as e:
            print(e)
    return X,Y,result_points


def load_automaticaly_measured_fetures(pths):
    X = []
    Y = []
    result_points = []
    for pth in pths:
        try:
            tg_id = os.path.splitext(os.path.basename(pth))[0][-len('Tg001-'):]
            print(tg_id)
            detected_json = json.load(open(pth))
            left = sorted([Point(y, x) for x, y in detected_json['left']], key=lambda a: a.y)
            right = sorted([Point(y, x) for x, y in detected_json['right']], key=lambda a: a.y)

            points = sorted(left+right, key=lambda p: p.y)
            points = points[1:-1]
            top, bot, pairs = createPairs(points)
            result_points.append((top, bot, pairs))
            feature_vec = to_feature_vector(top, bot, pairs)
            #feature_vec = to_euclideanfeature_vector(top, bot, pairs)
            X.append(feature_vec)
            Y.append(tg_id)
        except Exception as e:
            print(e)
    return X,Y,result_points

def evaluate_classification(X, Y, ignored=[], fig_prefix=''):
    cnt = 0
    x=[]
    y=[]

    pairs_labels = []
    color_map = {}
    colors=[]
    marker_map = {}
    markers = []
    default_color =(0.79,1.00,0.90)
    default_marker = 'o'

    for i, tg_id in enumerate(Y):
        try:
            print(tg_id)
            if 'a' in tg_id or 'b' in tg_id:  # TODO regex
                p_id = tg_id[:-1]
                Y[i] = p_id # modify pair names to match
                y.append(p_id)
                x.append(X[i])

                if not p_id in color_map:
                    color_map[p_id] = random_color()

                if not p_id in marker_map:
                    cnt+=1
                    marker_map[p_id] = get_shape(random.randint(0,1000))

                pairs_labels.append(tg_id)
                colors.append(color_map[p_id])
                markers.append(marker_map[p_id])
            else:
                pairs_labels.append(None)
                colors.append(default_color)
                markers.append(default_marker)

        except Exception as e:
            print(e)

    y = np.array(y)

    automated_pairs = filter_only_pairs(y)
    automated_pairs = np.unique(automated_pairs)
    print(fig_prefix + " pairs count: " + str(int(len(automated_pairs))))

    ks = [2,3,4,5,11,16,31]
    final_hits = []
    avg_rank = []
    mod_rank = []
    med_rank = []

    for k in ks:
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(x,y)
        y_pred  = knn.kneighbors(x,return_distance=False)
        hits = countHits(y, y, y_pred)
        hit_count = np.count_nonzero(hits > 0)
        hits = hits[hits > 0]
        final_hits.append(hit_count/y.size)

        if hits.size > 0:
            avg_rank.append(np.average(hits))
            med_rank.append(np.median(hits))
            mod_rank.append(sum(hits))
        else:
            avg_rank.append(0)
            med_rank.append(0)
            mod_rank.append(0)


    ks = [str(k-1) for k in ks]
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,9))
    ax[0].bar(ks, final_hits,width=0.9)
    ax[0].set_title('Nalezeno %')
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    for i, v in enumerate(final_hits):
        ax[0].text(i-0.2,v+0.01, int(100*v), color='black',fontsize=19,fontweight='bold' )

    ax[1].bar(ks, avg_rank,width=0.9)
    ax[1].set_title('Průměrné pořadí')
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    for i, v in enumerate(avg_rank):
        ax[1].text(i-0.2,v+0.01, "{:.1f}".format(v), color='black',fontsize=19,fontweight='bold')


    fig.savefig('figs/'+str(fig_prefix)+'_hits.png', bbox_inches='tight')
    plt.show()
    plt.clf()

    transformer = PCA(n_components=2)
    X_transformed = transformer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title('Pca projekce měření')
    for i, [x,y] in enumerate(X_transformed):
        ax.scatter(x,y, c=colors[i], marker=markers[i])
    step = 0.025

    plt.xlim(-0.1, 0.1)
    plt.ylim(-0.1, 0.1)

    for x_pos,y_pos,label in zip(X_transformed[:,0],X_transformed[:,1],pairs_labels):
        if label:
            ax.annotate(label,  # The label for this point
                        xy=(x_pos, y_pos),  # Position of the corresponding point
                        xytext=(3, 0),  # Offset text by 7 points to the right
                        textcoords='offset points',  # tell it to use offset points
                        ha='left',  # Horizontally aligned to the left
                        va='center',
                        fontsize= 'x-small'
                        )  # Vertical alignment is centered


    fig.savefig('figs/'+str(fig_prefix)+'_pca_scatter.png', bbox_inches='tight')
    plt.show()
    plt.clf()

def to_dict(X,Y):
    ret = {}
    for i,y in enumerate(Y):
        ret[y]=X[i]
    return ret

def evalutate_pixels_diferences(automated_points, Y_automated, manual_points, Y_manual):
    automated_dict = to_dict(automated_points, Y_automated)
    manual_dict = to_dict(manual_points, Y_manual)
    X_a_filter = []
    X_m_filter = []
    for k in automated_dict.keys():
        if k in manual_dict:
            X_a_filter.append(automated_dict[k])
            X_m_filter.append(manual_dict[k])

    calculate_pixel_histograms(X_a_filter, X_m_filter)

def evalutate_relative_diferences(X_automated, Y_automated, X_manual, Y_manual):
        automated_dict = to_dict(X_automated,Y_automated)
        manual_dict = to_dict(X_manual, Y_manual)
        X_a_filter = []
        X_m_filter = []
        for k in automated_dict.keys():
            if k in manual_dict:
                X_a_filter.append(automated_dict[k])
                X_m_filter.append(manual_dict[k])

        calculate_histograms(X_a_filter, X_m_filter)

def calculate_histograms(X_a, X_m):

    names = ['l1', 'l2', ' l3', ' l4', ' l5', ' l6', ' r1', ' r2', ' r3', ' r4', ' r5', ' r6', ' d1', ' d2', ' d3',
             ' d4', ' d5']
    for j in range(17):
        dist =[ abs(X_a[i][j]-X_m[i][j]) for i in range(len(X_a)) ]
        create_histogram(dist, names[j])


def calculate_pixel_histograms(X_a, X_m):
    top_dist = [ X_a[i][0].distance(X_m[i][0])  for i in range(len(X_a))]
    create_histogram(top_dist, 'J_0')
    bot_dist = [X_a[i][1].distance(X_m[i][1]) for i in range(len(X_a))]
    create_histogram(bot_dist, 'J_6')

    for j in range(0,5):
        l_dist = [X_a[i][2][j].left.distance(X_m[i][2][j].left) for i in range(len(X_a))]
        create_histogram(l_dist, 'Jl_' + str(j))
        r_dist = [X_a[i][2][j].right.distance(X_m[i][2][j].right) for i in range(len(X_a))]
        create_histogram(r_dist, 'Jr_' + str(j))



def create_histogram(distances, name, bins=5):
    distances = np.array(distances)
    filtered = distances[~is_outlier(distances, thresh=1.0)]

    fig, ax = plt.subplots()
    ax.hist(filtered, bins=bins)
    ax.set_ylabel('Počet')
    ax.set_xlabel('Odchylka (px)')
    ax.set_title('Histogram odchylek spoje: ' + name)

    fig.savefig('figs/hists/hist_' + name + '.png', bbox_inches='tight')
    plt.clf()


def is_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    print(dirname)
    pths = glob.glob(dirname+'/../../detected_junctions/*.json')
    X_automated,Y_automated,automated_points = load_automaticaly_measured_fetures(pths)
    pths = glob.glob(dirname+'/../../detected_plastrons/test_labels_*.json')
    X_manual, Y_manual,manual_points  = load_hand_measured_fetures(pths)


    evaluate_classification(X_manual, Y_manual, fig_prefix='manual')
    evaluate_classification(X_automated, Y_automated, fig_prefix='automated')
    evalutate_relative_diferences(X_automated, Y_automated, X_manual, Y_manual)
    evalutate_pixels_diferences(automated_points, Y_automated, manual_points, Y_manual)







