# distutils: language = c++
# cython: profile=True
from collections import Counter, namedtuple

import cynetworkx as nx
import cython
cimport  numpy as np
import numpy as np
import math
from libcpp.utility cimport pair
from libcpp.list cimport list as cpplist
from libcpp cimport bool

c_fib_cache = {}

def  c_fibonacci(n):
    if n in c_fib_cache:
        return c_fib_cache[n]
    c_fib_cache[n] = fibonacci(n)
    return c_fib_cache[n]

def fibonacci(int n):
    if n<0:
        return 0
    elif n==1:
        return 0
    elif n==2:
        return 1
    else:
        return fibonacci(n-1)+fibonacci(n-2)

def dist_matej(yc,y,xc,x):
    return abs(xc - x)**2 + (1+abs(yc - y)) ** 2

def dist_matej_x(yc,y,xc,x):
    return abs(yc - y) + (1 + abs(xc - x)) ** 2

def dist_l2(yc,y,xc,x):
    return math.sqrt((yc - y)**2 + (xc - x)**2)

def dist_l2_fib(yc,y,xc,x):
    return

def dist_2_power_l2(yc,y,xc,x):
    return 2 ** dist_l2(yc,y,xc,x)

def dist_chebishev(yc,y,xc,x):
    return max(abs(yc-y),abs(xc-x))

def dist_one(yc,y,xc,x):
    return 1

def dist_piramid(yc,y,xc,x):
    cheb = dist_chebishev(yc,y,xc,x)
    return (cheb)**2

def dist_fibonacci_piramid(yc,y,xc,x):
    return c_fibonacci(int(dist_chebishev(yc,y,xc,x))+2)

def dist_fibonacci_piramid_sqr(yc,y,xc,x):
    return c_fibonacci(int(dist_chebishev(yc,y,xc,x))+2)**2

def dist_diff(yc,y,xc,x):
    return dist_chebishev(yc,y,xc,x) + ((abs(xc-x)-abs(yc-y))**2)

def  s_dist_mat(size, func):
    return dist_mat(size, size, func)



def rect_dist_mat(size, func):
    width = math.floor(size/2)+1
    return dist_mat(size, width, func)

def dist_mat(height, width, func):
    cdef int xc = 0;
    cdef int yc = math.floor(height / 2)
    cdef int[:,:] ret = np.zeros((height, width), dtype=int)
    for y in range(0,height):
        for x in range(0,width):
            ret[y][x]=int(func(yc,y,xc,x))
    ret[yc][xc]=-1
    return ret

size = 9
matej_weights = rect_dist_mat(size, dist_matej)
matej_x_weights = rect_dist_mat(size, dist_matej_x)
l2_weights = rect_dist_mat(size, dist_l2)
pow_weights = rect_dist_mat(size, dist_2_power_l2)
fibbo_weights = rect_dist_mat(size, dist_fibonacci_piramid)
fibbo_sqr_weights = rect_dist_mat(size, dist_fibonacci_piramid_sqr)
diff_weights = rect_dist_mat(size,dist_diff)
cheb_weighs =  rect_dist_mat(size, dist_chebishev)
ones =  rect_dist_mat(size, dist_one)

weights = rect_dist_mat(size, dist_piramid)


def tnp(tpl):
    return np.array(tpl).astype(np.int)


def identityBuilder(self,g):
    def identityGraph(img, weights=None):
        return g


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def binaryArrayIntoOrientedRedundantGraph(double[:,:] img, int[:,:] weights=None):

    graph = nx.DiGraph()

    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int m_i = math.floor( weights.shape[0]/2)
    cdef int m_j = weights.shape[1]
    cdef int ymin = 0
    cdef int ymax = 0
    cdef int xmin = 0
    cdef int xmax = 0
    cdef int weight = 0
    cdef int h,w,th,tw,wy,x,y = 0

    for h,w in np.transpose(np.nonzero(img)):
        ymin = max(0 , h-m_i)
        ymax = min(height, h+m_i+1)
        xmin = w
        xmax = min(width, w + m_j)
        masked = img[ymin:ymax, xmin:xmax]
        for y, x in np.transpose(np.nonzero(masked)):
            th = ymin + y
            tw = xmin + x
            wy = ymin - h + m_i + y
            weight = weights[wy][x]
            #print(str(img[h][w])+'->'+str(img[th][tw]) + '=' +str(weight))
            if weight > -1:
                graph.add_edge((h, w), (th, tw), weight=weight)
    return graph


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def binaryArrayIntoOrientedGraph(double[:,:] img, weights=None):
    #iteration graph
    ig = create_iteration_graph(weights)

    cdef int height = img.shape[0]
    cdef int width = img.shape[1]

    cdef int h_center = math.floor(weights.shape[0] / 2)

    graph = nx.DiGraph()

    cdef int h
    cdef int w
    cdef int i
    cdef int j
    cdef int ii, iii
    cdef int jj, jjj
    cdef double value
    cdef int weight
    cdef int th
    cdef int tw
    #cdef list[(int,int)] q
    cdef cpplist[pair[int,int]] q
    cdef pair[int,int] c_point
    cdef bool[:,:] removed

    counter = 0

    for h,w in np.transpose(np.nonzero(img)):
        value = img[h][w]
        if value > 0:
            removed = np.zeros(weights.shape, dtype=np.bool)
            q.push_back(pair[int,int](h_center, 0))
            while not q.empty():
                c_point = q.front()
                q.pop_front()
                for (i, j) in ig.successors(c_point):
                    if not removed[i][j]:
                        weight = weights[i][j]
                        th = h + i - h_center
                        tw = w + j
                        if th < height and th >= 0 and tw < width:
                            if weight > 0 and img[th][tw] > 0:
                                graph.add_edge((h, w), (th, tw), weight=weight)
                                for (ii, jj), succ in nx.bfs_successors(ig, (i, j)):
                                    if not removed[ii][jj]:
                                        for iii, jjj in succ:
                                            removed[iii][jjj] = True
                            else:
                                if not removed[i][j]:
                                    q.push_back(pair[int,int](i, j))

    return graph

def create_iteration_graph(_weights, graph_builder=binaryArrayIntoOrientedRedundantGraph):
    mask = np.ones(_weights.shape)
    min_path, min_len, _, g, _ = gads(mask, graph_builder=graph_builder, weights=_weights)
    h_center = math.floor(_weights.shape[0] / 2)

    out_g = nx.DiGraph()
    paths = []
    for node in g:
        out_g.add_node(node)
        try:
            p = nx.all_shortest_paths(g, (h_center, 0), node, weight='weight')
            for pp in p:
                paths.append(pp)
        except  nx.exception.NetworkXNoPath as e:
            continue

    for v in paths:
        for i in range(len(v)-1,0,-1):
            ed = g.get_edge_data(v[i-1], v[i])
            out_g.add_edge(v[i-1], v[i], **ed)
    return out_g

Point2D = namedtuple('Point2D', 'i_pos w_pos parent')

def buildBFSGraph(img, weights):
    height = img.shape[0]
    width = img.shape[1]
    sw_y = math.floor(weights.shape[0] / 2)
    sw_x = 0
    mw_y = weights.shape[0]
    mw_x = weights.shape[1]
    w_start_point=(sw_y,sw_x)
    visited = np.zeros(img.shape, dtype=np.bool)

    graph = nx.DiGraph()

    directions = [(1, 1), (0, 1), (-1, 1),]

    frontier = []
    for i in np.nonzero(img[:, 0]):
        frontier.append(Point2D(i_pos=(i[0],0),w_pos=w_start_point, parent=(i[0],0)))
    while frontier:
        s = frontier.pop()
        parent = s.parent
        nw_pos = s.w_pos
        vis = visited[s.i_pos]
        if img[s.i_pos] > 0:
            w = weights[s.w_pos[0]][s.w_pos[1]]
            if w > -1:
                graph.add_edge(parent,s.i_pos, weight=w )
                parent=s.i_pos
                nw_pos=(sw_y,sw_x)
                visited[s.i_pos] = True

        if not vis:
            for d in directions:
                ni_pos = (s.i_pos[0]+d[0],s.i_pos[1]+d[1])
                nnw_pos = (nw_pos[0] + d[0], nw_pos[1] + d[1])

                if height > ni_pos[0] >= 0 \
                and width > ni_pos[1] >= 0 \
                and mw_y > nnw_pos[0] >= 0 \
                and mw_x > nnw_pos[1] >= 0 :
                    p = Point2D(i_pos=ni_pos, w_pos=nnw_pos, parent=parent)
                    #print('adding: ' + str(ni_pos) + ' v:' + str(img[ni_pos]))
                    frontier.append(p)

    return graph

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def binaryArrayIntoOptimalGraph(double[:,:] img,int[:,:] weights=None):
    cdef int height = img.shape[0]
    cdef int width = img.shape[1]

    graph = nx.DiGraph()

    cdef int max_weight = 5000
    cdef int min_weight = 1

    cdef int h = 0
    cdef int w = 0
    cdef int weight = 1000
    cdef bool go_left = False
    for h in range(0, height):
        for w in range(0, width):
            go_left = w + 1 < width
            if h - 1 >= 0:
                if go_left:
                    graph.add_edge((h,w),(h-1,w+1),weight=min_weight if img[h-1][w+1] > 0 else max_weight)
                graph.add_edge((h,w),(h-1,w),weight=min_weight if img[h-1][w] > 0 else max_weight)
            if h + 1 < height:
                if go_left:
                   graph.add_edge((h,w),(h+1,w+1),weight=min_weight if img[h+1][w+1] > 0 else max_weight)
                graph.add_edge((h,w),(h+1,w),weight=min_weight if img[h+1][w] > 0 else max_weight)
            if go_left:
                 graph.add_edge((h,w),(h,w+1),weight=min_weight if img[h][w+1] > 0 else max_weight)
    return graph

def path_cost(G, path):
    return sum(
        [G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1)])

def gads(
    img,
    start_width=None,
    end_width=None,
    start_nodes=None,
    end_nodes=None,
    graph_builder=binaryArrayIntoOrientedGraph,
    weights=None
):
    h = img.shape[0]
    w = img.shape[1]

    if end_width is None:
        end_width = w - 1
    else:
        end_width = w - end_width

    if start_width is None:
        start_width = 0

    if start_nodes is None:
        start_nodes = [(ht, start_width) for ht in range(0, h) if
                       img[ht][start_width] > 0]

    if len(start_nodes) < 1:
        raise Exception('No starting points')

    if end_nodes is None:
        end_nodes = \
            set([(ht, end_width) for ht in range(0, h) if
                 img[ht][end_width] > 0])

    if len(start_nodes) < 1:
        raise Exception('No end_nodes points')

    g = graph_builder(img, weights=weights)

    min_len = 99999999999
    min_path = None
    start_nodes_n = [n for n in start_nodes if n in g]
    distances, paths = nx.multi_source_dijkstra(g, start_nodes_n, weight='weight')

    for end in end_nodes:
        if (end in distances):
            cost = distances[end]
            if cost < min_len:
                min_len = cost
                min_path = paths[end]

    ret = np.zeros(img.shape)
    if min_path is not None:
        for c in min_path:
            ret[c] = 1

    all = np.zeros(img.shape)
    for p, path in paths.items():
        if p in end_nodes:
            for c in path:
                all[c] = all[c] + 1

    return min_path, min_len, ret, g, all

