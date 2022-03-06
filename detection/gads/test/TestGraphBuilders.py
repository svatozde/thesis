from unittest import TestCase

import GADS
import math
import random
import time
import cynetworkx as nx
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte, segmentation, filters, color
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.graph import shortest_path
from skimage.io import imsave, imread
from skimage.measure import label
from skimage.morphology import thin
from sklearn.feature_extraction.image import _to_graph, img_to_graph
from skimage.filters import meijering, threshold_mean


class TestGraphBuilders(TestCase):

    def endge_in_path(self, min_path, u, v):
        if min_path:
            for i in range(0, len(min_path) - 1):
                s = min_path[i]
                t = min_path[i + 1]
                if u == s and v == t:
                    return True
        return False


    def draw_graph(self, g, min_path, figure_name=None, show_labels=False):
        color_map = []
        node_size = []

        for node in g:
            if min_path and node in min_path:
                color_map.append('red')
                node_size.append(20)
            else:
                color_map.append('green')
                node_size.append(17)

        edge_color_map = []
        eddge_width = []
        for u, v, d in g.edges(data=True):
            if self.endge_in_path(min_path, u, v):
                edge_color_map.append('red')
                eddge_width.append(1)
            else:
                edge_color_map.append('black')
                eddge_width.append(1)

        pos = {p: (p[1], p[0]) for p in g}
        nx.draw(
            g,
            cmap=plt.get_cmap('jet'),
            node_color=color_map,
            node_shape='s',
            node_size=node_size,
            with_labels=False,
            pos=pos,
            arrows=True,
            arrowsize=10,
            arrowstyle='-|>',
            width=eddge_width
        )

        if show_labels:
            edge_labels = {(u, v): g[u][v]['weight'] for u, v in g.edges}

            nx.draw_cynetworkx_edge_labels(
                g,
                pos,
               edge_labels=edge_labels,
                font_size=8,
            )
        if figure_name:
            plt.savefig(figure_name)
        plt.show()
        plt.clf()


    def overlapCout(self, a, b):
        out = np.logical_and(a, b)
        return np.count_nonzero(out)


    def inverseOverlapCout(self, a, b):
        ia = np.max(a) - a
        ib = np.max(b) - b
        return self.overlapCout(ia, ib)


    def tpr(self, a, b):
        nz_b = max(1, np.count_nonzero(b))
        same = self.overlapCout(a, b)

        return same / nz_b


    def accuracy(self,a, b):
        TP = self.overlapCout(a, b)
        TN = self.inverseOverlapCout(a, b)
        return (TP + TN) / a.size


    def identity(self,g):
        def identityGraph(img, weights=None):
            return g

        return identityGraph


    def test_performance(self):
        reps =2
        for (w,h) in [(2000,500)]:
            for mat_size in [7]:
                weights = GADS.s_dist_mat(mat_size, GADS.dist_piramid)
                # weights = GADS.penMat(mat_size, GADS.dist_matej)
                for gap in [0.3]:
                    for density in [0.05]:
                        for test in range(0, 20):
                            mask = np.zeros((h, w))
                            for i in range(w):
                                ss = math.sin((7 * i) / w)
                                hi = int((h / 2) + (ss * (h / 6)))
                                if random.random() > gap:
                                    mask[hi, i] = 255

                            original = np.copy(mask)

                            for i in range(int(w / 9), int(8 * w / 9)):
                                ss = math.cos((28 * i) / w)
                                hi = int((h / 5) + (ss * (h / 20)))
                                if random.random() > gap:
                                    mask[hi, i] = 255

                                ss = math.cos((28 * i) / w)
                                hi = int((h - (h / 5)) + (ss * (h / 20)))
                                if random.random() > gap:
                                    mask[hi, i] = 255

                            for iw in range(w):
                                for ih in range(h):
                                    if random.random() > (1-density):
                                        mask[ih, iw] = 255

                            node_count = np.count_nonzero(mask)
                            imsave('figs/a_mask_noised_' + str(time.time()) + '_mask.png', mask.astype(np.uint8))

                            # print('------------ test ' + str(test) + ' ------------')
                            # print('node count: ' + str(node_count))
                            res = np.zeros(mask.shape)
                            pth, c = shortest_path(mask, axis=0, reach=5, output_indexlist=True)
                            self.drawNodes(res, pth)

                            start = time.time()
                            nmin_path, nmin_len, ret_new, ng, alln = GADS.gads(
                                mask,
                                graph_builder=GADS.binaryArrayIntoOrientedGraph,
                                weights=weights
                            )
                            end = time.time()
                            new_time = end - start
                            start = time.time()
                            for _ in range(0,reps):
                                GADS.gads(
                                    mask,
                                    graph_builder=self.identity(ng),
                                    weights=weights
                                )

                            end = time.time()
                            new_d_time = end - start
                            new_edges = ng.number_of_edges()

                            start = time.time()
                            min_path, min_len, ret_old, rg, all_old = GADS.gads(
                                mask,
                                graph_builder=GADS.binaryArrayIntoOrientedRedundantGraph,
                                weights=weights
                            )
                            end = time.time()
                            red_time = end - start

                            start = time.time()
                            for _ in range(0, reps):
                                GADS.gads(
                                    mask,
                                    graph_builder=self.identity(rg),
                                    weights=weights
                                )

                            end = time.time()
                            d_time_red = end-start

                            tpr_same = self.tpr(ret_new, ret_old)

                            tpr_opt = self.tpr(ret_old, original)

                            tpr_new = self.tpr(ret_new, original)

                            print(
                                'res: ' + str((w,h)) +
                                ' m: ' + str(mat_size) +
                                ' opt: ' + str("{:.4f}".format(red_time)) +
                                ' d_opt: ' + str("{:.4f}".format(d_time_red)) +
                                ' new: ' + str("{:.4f}".format(new_time)) +
                                ' d_new: ' + str("{:.4f}".format(new_d_time)) +
                                ' e_ratio: ' + str("{:.4f}".format(new_edges / rg.number_of_edges())) +
                                ' opt_edges: ' + str(rg.number_of_edges()) +
                                ' new_edges: ' + str(new_edges) +
                                # ' opt_l: ' + str(min_len) +
                                # ' new_l: ' + str(new_l) +
                                ' similarity: ' + str("{:.4f}".format(tpr_same)) +
                                ' new_tpr:' + str("{:.4f}".format(tpr_new)) +
                                ' opt_tpr:' + str("{:.4f}".format(tpr_opt)) +
                                ' node_count: ' + str(np.count_nonzero(mask)) +
                                ' density: ' + str("{:.4f}".format(np.count_nonzero(mask) / mask.size)) +
                                ' gap: ' + str("{:.4f}".format(gap))
                            )

                            #ret_new = (ret_new * 255)
                            #ret_old = (ret_old * 255)
                            imsave('figs/new_gads_p_' + str(time.time()) + '_new_path_.jpg', ret_new.astype(np.uint8),
                                   check_contrast=False)
                            imsave('figs/opt_gads_p_' + str(time.time()) + '_old_path.jpg', ret_old.astype(np.uint8),
                                  check_contrast=False)
                            # draw_graph(ng, nmin_path, figure_name='figs/gads_g_' + str(test) + '_new_gg_.png')
                            # draw_graph(g, min_path, figure_name='figs/gads_g_' + str(test) + '_old_gg_.png')


    def test_BfsBuilder(self):
        mask = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ).astype(np.double)
        t_weights = GADS.s_dist_mat(5, GADS.dist_matej)

        g = GADS.buildBFSGraph(mask, t_weights)
        self.draw_graph(g,[])

        min_path, min_len, ret_old, rg, _ = GADS.gads(
            mask,
            graph_builder=GADS.binaryArrayIntoOrientedRedundantGraph,
            weights=t_weights
        )

        self.draw_graph(rg, min_path)

        min_path, min_len, ret_old, ng, _ = GADS.gads(
            mask,
            graph_builder=GADS.binaryArrayIntoOrientedGraph,
            weights=t_weights
        )


        self.draw_graph(ng, min_path)

        print(g.number_of_edges())
        print(rg.number_of_edges())
        print(ng.number_of_edges())
        self.assertGreater(rg.number_of_edges(),g.number_of_edges())




    def test_redundancy(self):
        mask = np.array(
            [
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            ]
        ).astype(np.double)

        t_weights =GADS.s_dist_mat(7, GADS.dist_matej)
        ig = GADS.create_iteration_graph(t_weights)
        self.draw_graph(ig, [])

        min_path, min_len, ret_old, g, _ = GADS.gads(
            mask,
            graph_builder=GADS.binaryArrayIntoOrientedGraph,
            weights=t_weights
        )

        print(g.number_of_edges())
        self.draw_graph(g, min_path, figure_name='figs/g_optimal.png')

        min_path, min_len_red, ret_old, g_red, _ = GADS.gads(
            mask,
            graph_builder=GADS.binaryArrayIntoOrientedRedundantGraph,
            weights=t_weights
        )

        print(g_red.number_of_edges())
        self.assertEquals(min_len, min_len_red)
        self.draw_graph(g_red, min_path, figure_name='figs/g_redundant.png')

        self.assertGreater(g_red.number_of_edges(), g.number_of_edges())

    def drawNodes(self,img, nodes):
        for i, j in nodes:
            img[i, j] = 1


    def calculate_complexity(self, g: nx.Graph):
        e = g.number_of_edges()
        v = g.number_of_nodes()
        return (v + e) * math.log(v)


    def test_on_large_leaf_picture(self):
        original = imread('leaf.jpg')
        gray = img_as_ubyte(rgb2gray(original))
        img = canny(
            gray,
            sigma=3,
            use_quantiles=True,
        )

        mimg = meijering(gray, sigmas=range(1,10,2),black_ridges=True)
        mimg = mimg > threshold_mean(mimg)
        mimg = np.invert(mimg)
        mimg = thin(mimg, max_iter=7)
        img = img + mimg




        result = np.zeros(img.shape)
        img = img.astype(np.double)

        h = img.shape[0]
        start_nodes = [(i, 10) for i, v in enumerate(img[:, 10]) if v > 0]
        self.drawNodes(result, start_nodes)
        # print('build start time:')
        # start = time.time()
        # g = GADS.binaryArrayIntoOptimalGraph(img, weights=None)
        # end = time.time()
        # print('build graph time: ' + str(end - start))
        # print('complexity: ' + str(calculate_complexity(g)))

        print('build old start time:')
        start = time.time()
        weights = GADS.rect_dist_mat(9, GADS.dist_piramid)
        g = GADS.binaryArrayIntoOrientedRedundantGraph(img, weights=weights)
        end = time.time()
        print('build old graph time: ' + str(end - start))
        print('complexity old : ' + str(self.calculate_complexity(g)))

        h = img.shape[0]
        w = img.shape[1]
        hi = int(h / 7)
        for i in range(0, h, hi):
            result[i:i + hi, w - 15]=1
            end_nodes = [(ii + i, w - 15) for ii, v in enumerate(img[i:i + hi, w - 15]) if v > 0]
            start = time.time()
            min_path, min_len, ret_old, g, _ = GADS.gads(
                img,
                graph_builder=self.identity(g),
                weights=weights,
                start_nodes=start_nodes,
                end_nodes=end_nodes
            )
            end = time.time()
            print('dijkstra time: ' + str(end - start))

            #

            np.putmask(result, ret_old, 255)
            np.putmask(gray, result, 255)
            imsave('figs/reconstructed_leaf_' + str(time.time()) + '_.png', result)

        wi = int(w / 10)
        sub = 100
        for i in range(int(w / 4), w, wi):
            result[sub, i:i + wi] = 1
            end_nodes = [(sub, i + ii) for ii, v in enumerate(img[sub, i:i + wi]) if v > 0]
            start = time.time()
            min_path, min_len, ret_old, g, _ = GADS.gads(
                img,
                graph_builder=self.identity(g),
                weights=weights,
                start_nodes=start_nodes,
                end_nodes=end_nodes
            )
            #drawNodes(result, end_nodes)
            end = time.time()
            print('dijkstra time: ' + str(end - start))
            result[h - sub, i:i + wi] = 1
            end_nodes = [(h - sub, ii + i) for ii, v in enumerate(img[h - sub, i:i + wi]) if v > 0]
            start = time.time()
            min_path, min_len, ret_old, g, _ = GADS.gads(
                img,
                graph_builder=self.identity(g),
                weights=weights,
                start_nodes=start_nodes,
                end_nodes=end_nodes
            )
            # drawNodes(result, end_nodes)
            end = time.time()
            print('dijkstra time: ' + str(end - start))

            np.putmask(result, ret_old, 255)
            np.putmask(gray, result, 255)
            imsave('figs/reconstructed_1_' + str(time.time()) + 'leaf_.png', gray)

        print()



    def test_show_grpahs_for_different_matirces(self):

        for weights, name in [
            ( GADS.rect_dist_mat(9,dist_one), ' ones'),
            (GADS.matej_x_weights, 'matej_x_weights'),
            (GADS.matej_weights, 'matej_weights'),
            (GADS.cheb_weighs, 'cheb_weighs'),
            (GADS.l2_weights, 'l2_weights'),
            (GADS.pow_weights, 'pow_weights'),
            (GADS.fibbo_weights, 'fibbo_weights'),
            (GADS.fibbo_sqr_weights, 'fibbo_sqr_weights'),
        ]:
            print(weights.base)
            ig = GADS.create_iteration_graph(weights)
            self.draw_graph(ig, [],figure_name='figs/'+str(name)+'.png')


    def test_label(self):
        mask = np.array(
            [
                [1, 0, 0, 0, 0, 1],
                [0, 0, 1, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 0],
                [1, 0, 0, 0, 0, 1]
            ]
        )
        conn = np.ones((3, 3))
        out_labels, num = label(mask, connectivity=1, return_num=True)
        print(out_labels)

        ret = np.zeros(mask.shape)
        for lab in range(1, num + 1):
            segment = np.argwhere(out_labels == lab)
            shp = segment.shape
            if shp[0] > 3:
                ret[[*segment.T]] = 1
        print(ret)


def dist_one(yc, y, xc, x):
    return 1