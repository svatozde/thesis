from shapely.geometry import Point
from unittest import TestCase

from evaluation.evaluation_features import PointPair


class TestPointPair(TestCase):

    def test_between_teo_points(self):
        top = Point(2,0)
        left = Point(1.5,1.1)
        right = Point(2.5, 1.2)
        bot = Point(2,2)
        pp = PointPair(left,right)

        r1 = pp.top_right_distnce(top)
        r2 = pp.bot_right_distnce(bot)
        r = r1 + r2
        l1 = pp.top_left_distnce(top)
        l2 = pp.bot_left_distnce(bot)
        l = l1 + l2
        self.assertEquals(l1+l2,r1+r2)


    def test_two_pointpairs(self):
        top = Point(1.9, 0)

        left_a = Point(1.9, 1.1)
        right_a = Point(2.1, 1.2)

        left_b = Point(1.9, 2.2)
        right_b = Point(2.1, 2.1)

        left_c = Point(1.9, 3.1)
        right_c = Point(2.1, 3.2)

        bot = Point(1.9, 4)

        pa = PointPair(left_a,right_a)
        pb = PointPair(left_b, right_b)
        pc = PointPair(left_c, right_c)

        r0 = pa.top_right_distnce(top)
        l0 = pa.top_left_distnce(top)

        r1 = pb.get_right_distnce(pa)
        l1 = pb.get_left_distnce(pa)

        r2 = pc.get_right_distnce(pb)
        l2 = pc.get_left_distnce(pb)

        r3 = pa.bot_right_distnce(bot)
        l3 = pa.bot_left_distnce(bot)

        r = r0 + r1 + r2 + r3
        l = l0 + l1 + l2 + l3

        self.assertEquals(r,l)