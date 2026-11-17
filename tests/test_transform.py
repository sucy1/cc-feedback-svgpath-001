import unittest
from math import cos, radians, sin, sqrt

import pytest

from svg.path import Arc, Close, CubicBezier, Line, Move, Path, QuadraticBezier
from svg.path import parse_path


def apply(matrix, point):
    a, b, c, d, e, f = matrix
    return complex(a * point.real + c * point.imag + e,
                   b * point.real + d * point.imag + f)


def rotation_matrix(angle):
    r = radians(angle)
    return (cos(r), sin(r), -sin(r), cos(r), 0, 0)


class TransformTest(unittest.TestCase):
    def test_translate_lines(self) -> None:
        path = parse_path("M 0 0 L 10 0 L 10 10 z")
        moved = path.transform((1, 0, 0, 1, 5, 7))
        self.assertEqual(moved[0], Move(5 + 7j))
        self.assertEqual(moved[1], Line(5 + 7j, 15 + 7j))
        self.assertEqual(moved[2], Line(15 + 7j, 15 + 17j))
        self.assertIsInstance(moved[3], Close)
        self.assertEqual(moved[3].start, 15 + 17j)
        self.assertEqual(moved[3].end, 5 + 7j)
        # The original path is unchanged.
        self.assertEqual(path[0], Move(0j))

    def test_scale_beziers(self) -> None:
        path = parse_path("M 1 2 C 3 4 5 6 7 8 Q 9 10 11 12")
        scaled = path.transform((2, 0, 0, 2, 0, 0))
        cubic = scaled[1]
        assert isinstance(cubic, CubicBezier)
        self.assertEqual(cubic.control1, 6 + 8j)
        self.assertEqual(cubic.control2, 10 + 12j)
        self.assertEqual(cubic.end, 14 + 16j)
        quad = scaled[2]
        assert isinstance(quad, QuadraticBezier)
        self.assertEqual(quad.control, 18 + 20j)
        # Uniform scaling by 2 doubles the length.
        self.assertAlmostEqual(scaled.length(), 2 * path.length())

    def test_rotate_cubic(self) -> None:
        path = parse_path("M 10 0 C 20 0 30 10 30 10")
        rotated = path.transform(rotation_matrix(90))
        for t in (0, 0.25, 0.5, 0.75, 1):
            expected = apply(rotation_matrix(90), path[1].point(t))
            self.assertAlmostEqual(rotated[1].point(t), expected)

    def test_smooth_and_relative_flags_survive(self) -> None:
        path = parse_path("m 1 1 c 1 0 2 1 3 1 s 2 1 3 1")
        moved = path.transform((1, 0, 0, 1, 10, 20))
        self.assertTrue(moved[0].relative)
        self.assertTrue(moved[2].smooth)
        # The smooth relation is preserved by the transform.
        self.assertTrue(moved[2].is_smooth_from(moved[1]))

    def test_invalid_matrix(self) -> None:
        path = parse_path("M 0 0 L 1 1")
        with pytest.raises(ValueError):
            path.transform((1, 0, 0, 1))

    def test_arc_uniform_scale_and_rotation(self) -> None:
        # A similarity transform maps parameter values 1:1.
        arc = Arc(0j, complex(10, 5), 30, True, True, 20 + 10j)
        path = Path(arc)
        angle = 45
        scale = 2.5
        r = radians(angle)
        matrix = (scale * cos(r), scale * sin(r),
                  -scale * sin(r), scale * cos(r), 3, -4)
        transformed = path.transform(matrix)
        new_arc = transformed[0]
        assert isinstance(new_arc, Arc)
        for t in (0, 0.25, 0.5, 0.75, 1):
            self.assertAlmostEqual(
                new_arc.point(t), apply(matrix, arc.point(t)), places=5
            )
        # The given radii are too small for the chord, so the arc carries a
        # radius_scale correction; the effective radii scale by `scale`.
        self.assertAlmostEqual(
            new_arc.radius.real, arc.radius.real * arc.radius_scale * scale
        )
        self.assertAlmostEqual(
            new_arc.radius.imag, arc.radius.imag * arc.radius_scale * scale
        )
        self.assertAlmostEqual(
            new_arc.length(), scale * arc.length(), places=6
        )

    def test_arc_non_uniform_scale(self) -> None:
        # Squashing a circle along x turns it into an axis-aligned ellipse.
        arc = Arc(-10 + 0j, complex(10, 10), 0, False, True, 10 + 0j)
        path = Path(arc)
        transformed = path.transform((2, 0, 0, 1, 0, 0))
        new_arc = transformed[0]
        assert isinstance(new_arc, Arc)
        self.assertAlmostEqual(new_arc.radius.real, 20)
        self.assertAlmostEqual(new_arc.radius.imag, 10)
        self.assertAlmostEqual(new_arc.rotation % 180, 0)
        self.assertAlmostEqual(new_arc.start, -20 + 0j)
        self.assertAlmostEqual(new_arc.end, 20 + 0j)

    def test_arc_skew_changes_rotation(self) -> None:
        # Under a shear the ellipse rotates; the transformed arc must still
        # pass through the transformed sample points of the original.
        arc = Arc(0j, complex(15, 8), 20, True, False, 30 + 10j)
        matrix = (1, 0.3, 0.5, 1, 0, 0)
        transformed = Path(arc).transform(matrix)
        new_arc = transformed[0]
        assert isinstance(new_arc, Arc)
        # Sample the original arc densely; every transformed sample must lie
        # on the new ellipse, and the new arc's bbox/length must match the
        # transformed samples.
        samples = [apply(matrix, arc.point(i / 2000.0)) for i in range(2001)]
        rx = new_arc.radius.real * new_arc.radius_scale
        ry = new_arc.radius.imag * new_arc.radius_scale
        cosr = cos(radians(new_arc.rotation))
        sinr = sin(radians(new_arc.rotation))
        center = new_arc.center
        for p in samples:
            # Rotate into the ellipse frame and check the ellipse equation.
            q = p - center
            u = cosr * q.real + sinr * q.imag
            v = -sinr * q.real + cosr * q.imag
            self.assertAlmostEqual((u / rx) ** 2 + (v / ry) ** 2, 1, places=6)
        xs = [p.real for p in samples]
        ys = [p.imag for p in samples]
        bbox = new_arc.boundingbox()
        self.assertAlmostEqual(bbox[0], min(xs), places=4)
        self.assertAlmostEqual(bbox[1], min(ys), places=4)
        self.assertAlmostEqual(bbox[2], max(xs), places=4)
        self.assertAlmostEqual(bbox[3], max(ys), places=4)

    def test_arc_reflection_flips_sweep(self) -> None:
        arc = Arc(0j, complex(10, 10), 0, True, True, 20 + 0j)
        mirrored = Path(arc).transform((-1, 0, 0, 1, 0, 0))
        new_arc = mirrored[0]
        assert isinstance(new_arc, Arc)
        self.assertFalse(new_arc.sweep)
        # The geometry is the mirror image of the original, traversed in the
        # same parameter direction (the flipped sweep compensates the
        # reflection).
        for t in (0, 0.25, 0.5, 0.75, 1):
            p = arc.point(t)
            self.assertAlmostEqual(new_arc.point(t), complex(-p.real, p.imag))

    def test_degenerate_arc(self) -> None:
        # Zero radius arcs are treated as lines and must not crash.
        arc = Arc(0j, complex(0, 0), 0, False, False, 10 + 5j)
        transformed = Path(arc).transform((2, 0, 0, 3, 1, 1))
        new_arc = transformed[0]
        assert isinstance(new_arc, Arc)
        self.assertEqual(new_arc.start, 1 + 1j)
        self.assertEqual(new_arc.end, 21 + 16j)

    def test_d_roundtrip(self) -> None:
        path = parse_path(
            "M 10 20 C 30 40 50 60 70 80 S 110 120 130 140 "
            "Q 150 160 170 180 T 210 220 "
            "A 30 15 45 1 0 250 260 L 300 300 Z"
        )
        matrix = (1.5, 0.5, -0.25, 2.0, 7, -3)
        transformed = path.transform(matrix)
        reparsed = parse_path(transformed.d())
        self.assertAlmostEqual(reparsed.length(), transformed.length(), places=4)
        for got, expected in zip(reparsed.boundingbox(), transformed.boundingbox()):
            self.assertAlmostEqual(got, expected, places=4)


class IsClosedTest(unittest.TestCase):
    def test_close_segment(self) -> None:
        self.assertTrue(parse_path("M 0 0 L 10 0 L 10 10 Z").is_closed())

    def test_open_path(self) -> None:
        self.assertFalse(parse_path("M 0 0 L 10 0 L 10 10").is_closed())

    def test_endpoints_coincide(self) -> None:
        self.assertTrue(parse_path("M 0 0 L 10 0 L 10 10 L 0 0").is_closed())

    def test_tolerance(self) -> None:
        path = parse_path("M 0 0 L 10 0 L 1e-7 0")
        self.assertTrue(path.is_closed())
        self.assertFalse(path.is_closed(tolerance=1e-9))

    def test_empty_path(self) -> None:
        self.assertFalse(Path().is_closed())

    def test_last_subpath_counts(self) -> None:
        self.assertTrue(parse_path("M 0 0 L 1 1 M 5 5 L 6 5 L 5 5").is_closed())
        self.assertFalse(parse_path("M 0 0 L 1 1 Z M 5 5 L 6 5").is_closed())


class FlattenTest(unittest.TestCase):
    def test_lines_only(self) -> None:
        path = parse_path("M 0 0 L 10 0 L 10 10 Z")
        self.assertEqual(
            path.flatten(), [[0j, 10 + 0j, 10 + 10j, 0j]]
        )

    def test_subpath_grouping(self) -> None:
        path = parse_path("M 0 0 L 1 1 M 5 5 L 6 6 L 7 7")
        subpaths = path.flatten()
        self.assertEqual(len(subpaths), 2)
        self.assertEqual(subpaths[0], [0j, 1 + 1j])
        self.assertEqual(subpaths[1], [5 + 5j, 6 + 6j, 7 + 7j])

    def test_quadratic(self) -> None:
        path = Path(QuadraticBezier(0j, 50 + 100j, 100 + 0j))
        (points,) = path.flatten(tolerance=0.1)
        self.assertEqual(points[0], 0j)
        self.assertEqual(points[-1], 100 + 0j)
        self.assertGreater(len(points), 4)
        self.assertLessEqual(self._max_deviation(path[0], points), 0.1)

    def test_cubic(self) -> None:
        path = Path(CubicBezier(0j, 0 + 100j, 100 - 100j, 100 + 0j))
        (points,) = path.flatten(tolerance=0.05)
        self.assertEqual(points[0], 0j)
        self.assertEqual(points[-1], 100 + 0j)
        self.assertLessEqual(self._max_deviation(path[0], points), 0.05)

    def test_arc(self) -> None:
        # A half circle of radius 100.
        arc = Arc(-100 + 0j, complex(100, 100), 0, False, True, 100 + 0j)
        (points,) = Path(arc).flatten(tolerance=0.1)
        self.assertEqual(points[0], -100 + 0j)
        self.assertEqual(points[-1], 100 + 0j)
        # Every point lies on the circle, and the sagitta stays in bounds.
        for p in points:
            self.assertAlmostEqual(abs(p), 100, places=6)
        self.assertLessEqual(self._max_deviation(arc, points), 0.1)
        # Chord error of a circle: r * (1 - cos(pi / n)); 64+ segments for
        # this tolerance, so the point count should be in a sane range.
        self.assertGreater(len(points), 30)

    def test_elliptical_arc(self) -> None:
        arc = Arc(0j, complex(80, 20), 30, True, False, 50 + 40j)
        (points,) = Path(arc).flatten(tolerance=0.2)
        self.assertEqual(points[0], arc.start)
        self.assertEqual(points[-1], arc.end)
        self.assertLessEqual(self._max_deviation(arc, points), 0.2)

    def test_flattened_length_matches(self) -> None:
        path = parse_path("M 0 0 C 30 60 70 -60 100 0 A 40 40 0 0 1 180 0")
        (points,) = path.flatten(tolerance=0.01)
        polyline_length = sum(
            abs(points[i + 1] - points[i]) for i in range(len(points) - 1)
        )
        self.assertAlmostEqual(
            polyline_length, path.length(), delta=path.length() * 0.01
        )

    def _max_deviation(self, segment, points) -> float:
        """Largest distance from densely sampled curve points to the polyline."""
        worst = 0.0
        for k in range(1001):
            p = segment.point(k / 1000.0)
            best = min(
                self._point_to_segment(p, points[i], points[i + 1])
                for i in range(len(points) - 1)
            )
            worst = max(worst, best)
        return worst

    @staticmethod
    def _point_to_segment(p, a, b) -> float:
        ab = b - a
        if abs(ab) < 1e-12:
            return abs(p - a)
        t = ((p - a).real * ab.real + (p - a).imag * ab.imag) / (abs(ab) ** 2)
        t = max(0.0, min(1.0, t))
        return abs(p - (a + ab * t))


if __name__ == "__main__":
    unittest.main()
