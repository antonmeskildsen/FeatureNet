import unittest
import math

import featurenet.helpers as helpers


class TestDistance(unittest.TestCase):
    known_distances = [
        # Simple cases
        (
            0, ((0, 0), (0, 0))
        ), (
            1, ((0, 0), (0, 1))
        ), (
            1, ((0, 0), (1, 0))
        ),
        # Float precision
        (
            math.sqrt(2), ((0, 0), (1, 1))
        ), (
            math.sqrt(353), ((2, 3), (10, 20))
        ),
        # Handling of negative numbers
        (
            0, ((-1, -1), (-1, -1))
        ), (
            1, ((-1, -1), (-1, 0))
        ), (
            1, ((-1, -1), (-1, -2))
        ), (
            math.sqrt(2), ((0, 0), (-1, -1))
        ),
    ]

    def test_known_distances(self):
        for result, points in self.known_distances:
            self.assertEqual(result, helpers.dist(*points))


class TestArea(unittest.TestCase):
    known_values = [
        (
            0, ((0, 0), (0, 0))
        ), (
            1, ((0, 0), (1, 1))
        ), (
            4, ((-1, -1), (1, 1))
        ), (
            1, ((-1, -1), (-2, -2))
        )
    ]

    def test_known_areas(self):
        for result, points in self.known_values:
            self.assertEqual(result, helpers.area(*points))


class TestBooleanOperations(unittest.TestCase):
    # Intersecting rectangle, IOU, Input rectangles
    known_values = [
        (
            ((0, 0), (0, 0)),
            0,
            (
                ((0, 0), (0, 0)),
                ((0, 0), (0, 0))
            )
        ),
        (
            ((0, 0), (1, 1)),
            1,
            (
                ((0, 0), (1, 1)),
                ((0, 0), (1, 1))
            )
        ),
        (
            ((0.5, 0), (1, 1)),
            0.5,
            (
                ((0, 0), (1, 1)),
                ((0.5, 0), (1, 1))
            )
        ),
        (
            ((0.5, 0.5), (1, 1)),
            0.25,
            (
                ((0, 0), (1, 1)),
                ((0.5, 0.5), (1, 1))
            )
        ),
        (
            ((0, 0), (0, 0)),
            0,
            (
                ((0, 0), (1, 1)),
                ((1, 1), (2, 2))
            )
        ),
    ]

    def test_known_intersects(self):
        for intersect, _, rectangles in self.known_values:
            self.assertEqual(intersect, helpers.intersect(*rectangles))

    def test_known_ious(self):
        for _, iou, rectangles in self.known_values:
            self.assertEqual(iou, helpers.iou(*rectangles))


class TestScoreFunctions(unittest.TestCase):
    invalid_values = [
        ((0, 0), (0, 0)),
        ((-1, 0), (0, 0)),
        ((0, -1), (0, 0)),
        ((0, 0), (-1, 0)),
        ((0, 0), (0, -1)),
    ]

    def test_invalid_input(self):
        for conf in self.invalid_values:
            self.assertRaises(ValueError, helpers.accuracy, conf)
            self.assertRaises(ValueError, helpers.precision, conf)
            self.assertRaises(ValueError, helpers.recall, conf)

    known_values = [
        # Conf,
        # accuracy, precision, recall, f1 score
        (
            ((1, 1), (1, 1)),
            0.5, 0.5, 0.5, 0.5
        ), (
            ((1, 0), (0, 0)),
            1, 1, 1, 1
        ), (
            ((0, 1), (0, 0)),
            0, 0, 0, 0
        ), (
            ((0, 0), (1, 0)),
            0, 0, 0, 0
        ), (
            ((0, 0), (0, 1)),
            1, 0, 0, 0
        ), (
            ((10, 20), (15,  7)),
            17/52, 10/30, 10/25, 4/11
        )
    ]

    def test_known_accuracies(self):
        for conf, accuracy, _, _, _ in self.known_values:
            self.assertEqual(accuracy, helpers.accuracy(conf))

    def test_known_precisions(self):
        for conf, _, precision, _, _ in self.known_values:
            self.assertEqual(precision, helpers.precision(conf))

    def test_known_recalls(self):
        for conf, _, _, recall, _ in self.known_values:
            self.assertEqual(recall, helpers.recall(conf))

    def test_known_f1_scores(self):
        for _, _, precision, recall, f1 in self.known_values:
            self.assertEqual(f1, helpers.f1_score(precision, recall))
