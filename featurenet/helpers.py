import math
import torch


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def area(p1, p2):
    return abs((p2[0] - p1[0]) * (p2[1] - p1[1]))


def intersect(ra, rb):
    pa1, pa2 = ra
    pb1, pb2 = rb

    if pb1[0] < pa2[0] and pb2[0] > pa1[0] \
            and pb1[1] < pa2[1] and pb2[1] > pa1[1]:

        p1 = (max(pa1[0], pb1[0]), max(pa1[1], pb1[1]))
        p2 = (min(pa2[0], pb2[0]), min(pa2[1], pb2[1]))
        return p1, p2
    else:
        return (0, 0), (0, 0)


def iou(ra, rb):
    pi1, pi2 = intersect(ra, rb)
    intersection_area = area(pi1, pi2)
    ra_area = area(*ra)
    rb_area = area(*rb)

    union_area = ra_area + rb_area - intersection_area

    if union_area == 0:
        return 0
    else:
        return intersection_area / union_area


def _validate_conf(conf):
    negative = (conf[0][0] < 0
                or conf[0][1] < 0
                or conf[1][0] < 0
                or conf[1][1] < 0)

    zero_sum = (conf[0][0] == 0
                and conf[0][1] == 0
                and conf[1][0] == 0
                and conf[1][1] == 0)

    if negative or zero_sum:
        raise ValueError("Input matrix is invalid (zero or negative elements)")


def accuracy(conf):
    _validate_conf(conf)
    return \
        (conf[0][0] + conf[1][1]) / (
                    conf[0][0] + conf[0][1] + conf[1][0] + conf[1][1])


def precision(conf):
    _validate_conf(conf)
    denom = conf[0][0] + conf[0][1]
    if denom == 0:
        return 0
    else:
        return conf[0][0] / denom


def recall(conf):
    _validate_conf(conf)
    denom = conf[0][0] + conf[1][0]
    if denom == 0:
        return 0
    return conf[0][0] / denom


def f1_score(precision, recall):
    recall_inverse = 1 / recall if recall != 0 else 0
    precision_inverse = 1 / precision if precision != 0 else 0
    k = recall_inverse + precision_inverse
    if k == 0:
        return 0
    else:
        return 2 / k


def gray_to_rgb(tensor):
    return torch.cat((tensor, tensor, tensor))
