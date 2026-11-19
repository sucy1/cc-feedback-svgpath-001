import math
import unittest

from svg.path import Arc, Close, CubicBezier, Line, Move, Path, QuadraticBezier
from svg.path import parse_path


def scale(sx, sy=None):
    if sy is None:
        sy = sx
    return (sx, 0, 0, sy, 0, 0)


def translate(tx, ty):
    return (1, 0, 0, 1, tx, ty)


def rotate(degrees):
    r = math.radians(degrees)
    return (math.cos(r), math.sin(r), -math.sin(r), math.cos(r), 0, 0)


def matmul(m1, m2):
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def apply(m, x, y):
    a, b, c, d, e, f = m
    return complex(a * x + c * y + e, b * x + d * y + f)


class TransformTestCase(unittest.TestCase):
    def assertPointsAlmostEqual(self, p1, p2, places=6):
        self.assertAlmostEqual(p1.real, p2.real, places=places)
        self.assertAlmostEqual(p1.imag, p2.imag, places=places)

    def test_line_translate(self):
        path = Path(Line(0 + 0j, 10 + 5j))
        moved = path.transform(translate(3, 4))
        self.assertEqual(moved[0].start, 3 + 4j)
        self.assertEqual(moved[0].end, 13 + 9j)
        # Original is unchanged
        self.assertEqual(path[0].start, 0 + 0j)

    def test_cubic_control_points(self):
        path = Path(CubicBezier(0j, 1 + 2j, 3 + 4j, 5 + 6j))
        scaled = path.transform(scale(2))
        seg = scaled[0]
        self.assertEqual(seg.control1, 2 + 4j)
        self.assertEqual(seg.control2, 6 + 8j)
        self.assertEqual(seg.end, 10 + 12j)
        # Length scales linearly with uniform scale
        self.assertAlmostEqual(scaled.length(), 2 * path.length(), places=6)

    def test_quadratic_control_points(self):
        path = Path(QuadraticBezier(0j, 1 + 1j, 2 + 0j))
        scaled = path.transform(scale(3, 1))
        self.assertEqual(scaled[0].control, 3 + 1j)
        self.assertEqual(scaled[0].end, 6 + 0j)

    def test_move_and_close(self):
        path = Path(Move(1 + 1j), Line(1 + 1j, 2 + 2j), Close(2 + 2j, 1 + 1j))
        moved = path.transform(translate(10, 20))
        self.assertEqual(moved[0].end, 11 + 21j)
        self.assertIsInstance(moved[2], Close)
        self.assertEqual(moved[2].end, 11 + 21j)

    def test_arc_uniform_scale_and_rotate(self):
        # A circle arc: scaling doubles the radius, rotation adds 30 degrees.
        arc = Arc(100 + 0j, 100 + 100j, 0, False, True, 0 + 100j)
        m = matmul(rotate(30), scale(2))
        result = arc_to_path(arc, m)
        new_arc = result[0]
        self.assertIsInstance(new_arc, Arc)
        self.assertAlmostEqual(new_arc.radius.real, 200, places=6)
        self.assertAlmostEqual(new_arc.radius.imag, 200, places=6)
        self.assertAlmostEqual(new_arc.rotation % 360, 30, places=6)
        self.assertEqual(new_arc.sweep, arc.sweep)
        self.assertEqual(new_arc.arc, arc.arc)

    def test_arc_nonuniform_scale(self):
        # Ellipse axis-aligned, scaled 2x in x: rx doubles, ry unchanged.
        arc = Arc(100 + 0j, 100 + 50j, 0, False, True, -100 + 0j)
        result = arc_to_path(arc, scale(2, 1))
        new_arc = result[0]
        self.assertAlmostEqual(new_arc.radius.real, 200, places=6)
        self.assertAlmostEqual(new_arc.radius.imag, 50, places=6)
        self.assertAlmostEqual(new_arc.rotation % 360, 0, places=6)

    def test_arc_geometry_preserved(self):
        # Sampled points of the transformed arc must equal the transformed
        # sampled points of the original arc, for a general affine matrix
        # including non-uniform scaling and rotation.
        arc = Arc(100 + 20j, 60 + 40j, 25, True, True, 10 + 90j)
        m = matmul(translate(7, -3), matmul(rotate(40), scale(1.5, 0.5)))
        new_arc = arc_to_path(arc, m)[0]
        for i in range(11):
            t = i / 10
            expected = apply(m, *xy(arc.point(t)))
            self.assertPointsAlmostEqual(new_arc.point(t), expected, places=5)

    def test_arc_mirror_flips_sweep(self):
        arc = Arc(100 + 0j, 50 + 30j, 15, True, True, 0 + 100j)
        mirror = (-1, 0, 0, 1, 0, 0)  # flip x
        new_arc = arc_to_path(arc, mirror)[0]
        self.assertEqual(new_arc.sweep, not arc.sweep)
        self.assertEqual(new_arc.arc, arc.arc)
        for i in range(11):
            t = i / 10
            expected = apply(mirror, *xy(arc.point(t)))
            self.assertPointsAlmostEqual(new_arc.point(t), expected, places=5)

    def test_arc_degenerate(self):
        # Zero-radius arc is a straight line; transform keeps endpoints.
        arc = Arc(0j, 0 + 0j, 0, False, True, 10 + 0j)
        result = arc_to_path(arc, scale(2))[0]
        self.assertPointsAlmostEqual(result.point(0.5), 10 + 0j)

    def test_d_roundtrip(self):
        path = parse_path(
            "M 10,20 L 30,40 C 1,2 3,4 5,6 Q 7,8 9,10 "
            "A 15,10 30 1,1 50,60 Z"
        )
        m = matmul(translate(5, 5), matmul(rotate(25), scale(2, 0.75)))
        transformed = path.transform(m)
        reparsed = parse_path(transformed.d())
        # Length is computed by numeric approximation, so allow a small
        # relative error between the two parameterizations.
        self.assertAlmostEqual(
            reparsed.length(), transformed.length(), delta=1e-3
        )
        # d() prints with %G (6 significant digits), so the re-parsed arc
        # parameterization differs slightly; allow for that.
        for b1, b2 in zip(reparsed.boundingbox(), transformed.boundingbox()):
            self.assertAlmostEqual(b1, b2, delta=1e-3)

    def test_bbox_matches_transformed_geometry(self):
        path = parse_path("M 0,0 C 10,20 30,40 50,0 A 20,10 0 0,1 80,20")
        m = matmul(rotate(15), scale(2, 3))
        transformed = path.transform(m)
        # Compare against the bbox of densely sampled transformed points.
        xs = []
        ys = []
        for i in range(501):
            p = path.point(i / 500)
            q = apply(m, p.real, p.imag)
            xs.append(q.real)
            ys.append(q.imag)
        bbox = transformed.boundingbox()
        self.assertAlmostEqual(bbox[0], min(xs), delta=0.5)
        self.assertAlmostEqual(bbox[1], min(ys), delta=0.5)
        self.assertAlmostEqual(bbox[2], max(xs), delta=0.5)
        self.assertAlmostEqual(bbox[3], max(ys), delta=0.5)


def xy(p):
    return p.real, p.imag


def arc_to_path(arc, matrix):
    return Path(arc).transform(matrix)


class IsClosedTestCase(unittest.TestCase):
    def test_close_segment(self):
        self.assertTrue(parse_path("M 0,0 L 10,0 L 10,10 Z").is_closed())

    def test_open_path(self):
        self.assertFalse(parse_path("M 0,0 L 10,0 L 10,10").is_closed())

    def test_coincident_endpoints(self):
        self.assertTrue(parse_path("M 0,0 L 10,0 L 0,0").is_closed())

    def test_tolerance(self):
        path = Path(Line(0j, 10 + 0j), Line(10 + 0j, 1e-8 + 0j))
        self.assertTrue(path.is_closed())
        self.assertFalse(path.is_closed(tolerance=1e-12))

    def test_empty_path(self):
        self.assertFalse(Path().is_closed())

    def test_close_in_middle_does_not_close_path(self):
        self.assertFalse(parse_path("M 0,0 L 1,0 Z L 5,5").is_closed())


class FlattenTestCase(unittest.TestCase):
    def test_line_only(self):
        path = Path(Line(0j, 10 + 0j), Line(10 + 0j, 10 + 10j))
        self.assertEqual(
            path.flatten(0.1), [[0j, 10 + 0j, 10 + 10j]]
        )

    def test_cubic_endpoints_and_tolerance(self):
        curve = CubicBezier(0j, 0 + 100j, 100 + 100j, 100 + 0j)
        (points,) = Path(curve).flatten(0.5)
        self.assertEqual(points[0], 0j)
        self.assertEqual(points[-1], 100 + 0j)
        # Every segment of the polyline stays within tolerance of the curve.
        self.assertLessEqual(max_deviation(curve, points), 0.5 + 1e-9)

    def test_quadratic(self):
        curve = QuadraticBezier(0j, 50 + 80j, 100 + 0j)
        (points,) = Path(curve).flatten(0.2)
        self.assertEqual(points[0], 0j)
        self.assertEqual(points[-1], 100 + 0j)
        self.assertLessEqual(max_deviation(curve, points), 0.2 + 1e-9)

    def test_arc(self):
        arc = Arc(100 + 0j, 50 + 50j, 0, True, True, 0 + 100j)
        (points,) = Path(arc).flatten(0.2)
        self.assertAlmostEqual(points[0].real, 100)
        self.assertAlmostEqual(points[-1].imag, 100)
        self.assertLessEqual(max_deviation(arc, points), 0.2 + 1e-9)
        # A quarter circle of radius 50 needs a reasonable number of points.
        self.assertGreater(len(points), 4)

    def test_subpaths_grouped_by_move(self):
        path = parse_path("M 0,0 L 1,0 M 5,5 L 6,6 L 7,7")
        subpaths = path.flatten(0.1)
        self.assertEqual(len(subpaths), 2)
        self.assertEqual(subpaths[0], [0j, 1 + 0j])
        self.assertEqual(subpaths[1], [5 + 5j, 6 + 6j, 7 + 7j])

    def test_close_appends_start_point(self):
        path = parse_path("M 0,0 L 10,0 L 10,10 Z")
        (points,) = path.flatten(0.1)
        self.assertEqual(points[0], points[-1])

    def test_discontinuity_starts_new_subpath(self):
        path = Path(Line(0j, 1 + 0j), Line(5 + 5j, 6 + 6j))
        self.assertEqual(len(path.flatten(0.1)), 2)


def max_deviation(segment, points):
    """Largest distance from the polyline to the sampled curve."""
    deviation = 0.0
    for i in range(1, 1000):
        p = segment.point(i / 1000)
        deviation = max(deviation, distance_to_polyline(p, points))
    return deviation


def distance_to_polyline(p, points):
    best = float("inf")
    for a, b in zip(points, points[1:]):
        chord = b - a
        if chord == 0:
            best = min(best, abs(p - a))
            continue
        t = ((p - a).real * chord.real + (p - a).imag * chord.imag) / abs(chord) ** 2
        t = min(1.0, max(0.0, t))
        best = min(best, abs(p - (a + chord * t)))
    return best


if __name__ == "__main__":
    unittest.main()
