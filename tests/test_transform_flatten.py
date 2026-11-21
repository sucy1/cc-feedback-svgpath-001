import math
import unittest
from typing import List, Optional, Tuple

from svg.path import Arc, Close, CubicBezier, Line, Move, Path, QuadraticBezier
from svg.path import parse_path
from svg.path.path import AffineMatrix, PathSegment


def scale(sx: float, sy: Optional[float] = None) -> AffineMatrix:
    if sy is None:
        sy = sx
    return (sx, 0, 0, sy, 0, 0)


def translate(tx: float, ty: float) -> AffineMatrix:
    return (1, 0, 0, 1, tx, ty)


def rotate(degrees: float) -> AffineMatrix:
    r = math.radians(degrees)
    return (math.cos(r), math.sin(r), -math.sin(r), math.cos(r), 0, 0)


def matmul(m1: AffineMatrix, m2: AffineMatrix) -> AffineMatrix:
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


def apply(m: AffineMatrix, x: float, y: float) -> complex:
    a, b, c, d, e, f = m
    return complex(a * x + c * y + e, b * x + d * y + f)


class TransformTestCase(unittest.TestCase):
    def assertPointsAlmostEqual(
        self, p1: complex, p2: complex, places: int = 6
    ) -> None:
        self.assertAlmostEqual(p1.real, p2.real, places=places)
        self.assertAlmostEqual(p1.imag, p2.imag, places=places)

    def test_line_translate(self) -> None:
        path = Path(Line(0 + 0j, 10 + 5j))
        moved = path.transform(translate(3, 4))
        self.assertEqual(moved[0].start, 3 + 4j)
        self.assertEqual(moved[0].end, 13 + 9j)
        # Original is unchanged
        self.assertEqual(path[0].start, 0 + 0j)

    def test_cubic_control_points(self) -> None:
        path = Path(CubicBezier(0j, 1 + 2j, 3 + 4j, 5 + 6j))
        scaled = path.transform(scale(2))
        seg = scaled[0]
        assert isinstance(seg, CubicBezier)
        self.assertEqual(seg.control1, 2 + 4j)
        self.assertEqual(seg.control2, 6 + 8j)
        self.assertEqual(seg.end, 10 + 12j)
        # Length scales linearly with uniform scale
        self.assertAlmostEqual(scaled.length(), 2 * path.length(), places=6)

    def test_quadratic_control_points(self) -> None:
        path = Path(QuadraticBezier(0j, 1 + 1j, 2 + 0j))
        scaled = path.transform(scale(3, 1))
        seg = scaled[0]
        assert isinstance(seg, QuadraticBezier)
        self.assertEqual(seg.control, 3 + 1j)
        self.assertEqual(seg.end, 6 + 0j)

    def test_move_and_close(self) -> None:
        path = Path(Move(1 + 1j), Line(1 + 1j, 2 + 2j), Close(2 + 2j, 1 + 1j))
        moved = path.transform(translate(10, 20))
        self.assertEqual(moved[0].end, 11 + 21j)
        self.assertIsInstance(moved[2], Close)
        self.assertEqual(moved[2].end, 11 + 21j)

    def test_arc_uniform_scale_and_rotate(self) -> None:
        # A circle arc: scaling doubles the radius, rotation adds 30 degrees.
        arc = Arc(100 + 0j, 100 + 100j, 0, False, True, 0 + 100j)
        m = matmul(rotate(30), scale(2))
        new_arc = arc_to_path(arc, m)[0]
        assert isinstance(new_arc, Arc)
        self.assertAlmostEqual(new_arc.radius.real, 200, places=6)
        self.assertAlmostEqual(new_arc.radius.imag, 200, places=6)
        self.assertAlmostEqual(new_arc.rotation % 360, 30, places=6)
        self.assertEqual(new_arc.sweep, arc.sweep)
        self.assertEqual(new_arc.arc, arc.arc)

    def test_arc_nonuniform_scale(self) -> None:
        # Ellipse axis-aligned, scaled 2x in x: rx doubles, ry unchanged.
        arc = Arc(100 + 0j, 100 + 50j, 0, False, True, -100 + 0j)
        new_arc = arc_to_path(arc, scale(2, 1))[0]
        assert isinstance(new_arc, Arc)
        self.assertAlmostEqual(new_arc.radius.real, 200, places=6)
        self.assertAlmostEqual(new_arc.radius.imag, 50, places=6)
        self.assertAlmostEqual(new_arc.rotation % 360, 0, places=6)

    def test_arc_geometry_preserved(self) -> None:
        # Sampled points of the transformed arc must equal the transformed
        # sampled points of the original arc, for a general affine matrix
        # including non-uniform scaling and rotation.
        arc = Arc(100 + 20j, 60 + 40j, 25, True, True, 10 + 90j)
        m = matmul(translate(7, -3), matmul(rotate(40), scale(1.5, 0.5)))
        new_arc = arc_to_path(arc, m)[0]
        assert isinstance(new_arc, Arc)
        for i in range(11):
            t = i / 10
            expected = apply(m, *xy(arc.point(t)))
            self.assertPointsAlmostEqual(new_arc.point(t), expected, places=5)

    def test_arc_mirror_flips_sweep(self) -> None:
        arc = Arc(100 + 0j, 50 + 30j, 15, True, True, 0 + 100j)
        mirror = (-1, 0, 0, 1, 0, 0)  # flip x
        new_arc = arc_to_path(arc, mirror)[0]
        assert isinstance(new_arc, Arc)
        self.assertEqual(new_arc.sweep, not arc.sweep)
        self.assertEqual(new_arc.arc, arc.arc)
        for i in range(11):
            t = i / 10
            expected = apply(mirror, *xy(arc.point(t)))
            self.assertPointsAlmostEqual(new_arc.point(t), expected, places=5)

    def test_arc_degenerate(self) -> None:
        # Zero-radius arc is a straight line; transform keeps endpoints.
        arc = Arc(0j, 0 + 0j, 0, False, True, 10 + 0j)
        result = arc_to_path(arc, scale(2))[0]
        self.assertPointsAlmostEqual(result.point(0.5), 10 + 0j)

    def test_d_roundtrip(self) -> None:
        path = parse_path(
            "M 10,20 L 30,40 C 1,2 3,4 5,6 Q 7,8 9,10 " "A 15,10 30 1,1 50,60 Z"
        )
        m = matmul(translate(5, 5), matmul(rotate(25), scale(2, 0.75)))
        transformed = path.transform(m)
        reparsed = parse_path(transformed.d())
        # Length is computed by numeric approximation, so allow a small
        # relative error between the two parameterizations.
        self.assertAlmostEqual(reparsed.length(), transformed.length(), delta=1e-3)
        # d() prints with %G (6 significant digits), so the re-parsed arc
        # parameterization differs slightly; allow for that.
        for b1, b2 in zip(reparsed.boundingbox(), transformed.boundingbox()):
            self.assertAlmostEqual(b1, b2, delta=1e-3)

    def test_bbox_matches_transformed_geometry(self) -> None:
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


def xy(p: complex) -> Tuple[float, float]:
    return p.real, p.imag


def arc_to_path(arc: Arc, matrix: AffineMatrix) -> Path:
    return Path(arc).transform(matrix)


class IsClosedTestCase(unittest.TestCase):
    def test_close_segment(self) -> None:
        self.assertTrue(parse_path("M 0,0 L 10,0 L 10,10 Z").is_closed())

    def test_open_path(self) -> None:
        self.assertFalse(parse_path("M 0,0 L 10,0 L 10,10").is_closed())

    def test_coincident_endpoints(self) -> None:
        self.assertTrue(parse_path("M 0,0 L 10,0 L 0,0").is_closed())

    def test_tolerance(self) -> None:
        path = Path(Line(0j, 10 + 0j), Line(10 + 0j, 1e-8 + 0j))
        self.assertTrue(path.is_closed())
        self.assertFalse(path.is_closed(tolerance=1e-12))

    def test_empty_path(self) -> None:
        self.assertFalse(Path().is_closed())

    def test_move_only_path_is_open(self) -> None:
        # A path without drawn geometry cannot be closed, even though its
        # single Move segment has identical start and end points.
        self.assertFalse(Path(Move(10 + 10j)).is_closed())
        self.assertFalse(parse_path("M 5,5 M 10,10").is_closed())

    def test_close_in_middle_does_not_close_path(self) -> None:
        self.assertFalse(parse_path("M 0,0 L 1,0 Z L 5,5").is_closed())


class FlattenTestCase(unittest.TestCase):
    def test_line_only(self) -> None:
        path = Path(Line(0j, 10 + 0j), Line(10 + 0j, 10 + 10j))
        self.assertEqual(path.flatten(0.1), [[0j, 10 + 0j, 10 + 10j]])

    def test_cubic_endpoints_and_tolerance(self) -> None:
        curve = CubicBezier(0j, 0 + 100j, 100 + 100j, 100 + 0j)
        (points,) = Path(curve).flatten(0.5)
        self.assertEqual(points[0], 0j)
        self.assertEqual(points[-1], 100 + 0j)
        # Every segment of the polyline stays within tolerance of the curve.
        self.assertLessEqual(max_deviation(curve, points), 0.5 + 1e-9)

    def test_quadratic(self) -> None:
        curve = QuadraticBezier(0j, 50 + 80j, 100 + 0j)
        (points,) = Path(curve).flatten(0.2)
        self.assertEqual(points[0], 0j)
        self.assertEqual(points[-1], 100 + 0j)
        self.assertLessEqual(max_deviation(curve, points), 0.2 + 1e-9)

    def test_arc(self) -> None:
        arc = Arc(100 + 0j, 50 + 50j, 0, True, True, 0 + 100j)
        (points,) = Path(arc).flatten(0.2)
        self.assertAlmostEqual(points[0].real, 100)
        self.assertAlmostEqual(points[-1].imag, 100)
        self.assertLessEqual(max_deviation(arc, points), 0.2 + 1e-9)
        # A quarter circle of radius 50 needs a reasonable number of points.
        self.assertGreater(len(points), 4)

    def test_subpaths_grouped_by_move(self) -> None:
        path = parse_path("M 0,0 L 1,0 M 5,5 L 6,6 L 7,7")
        subpaths = path.flatten(0.1)
        self.assertEqual(len(subpaths), 2)
        self.assertEqual(subpaths[0], [0j, 1 + 0j])
        self.assertEqual(subpaths[1], [5 + 5j, 6 + 6j, 7 + 7j])

    def test_close_appends_start_point(self) -> None:
        path = parse_path("M 0,0 L 10,0 L 10,10 Z")
        (points,) = path.flatten(0.1)
        self.assertEqual(points[0], points[-1])

    def test_discontinuity_starts_new_subpath(self) -> None:
        path = Path(Line(0j, 1 + 0j), Line(5 + 5j, 6 + 6j))
        self.assertEqual(len(path.flatten(0.1)), 2)

    def test_s_shaped_cubic_is_subdivided(self) -> None:
        # Regression test: the midpoint of this S-shaped cubic lies exactly
        # on the chord between its endpoints, while the curve bulges out by
        # ~28.87 units. A midpoint-only flatness test accepts it as a
        # straight line; the control-point based test must subdivide it.
        path = parse_path("M0 0 C 100 100 100 -100 200 0")
        (points,) = path.flatten(0.1)
        self.assertGreater(len(points), 2)
        self.assertLessEqual(max_deviation(path[1], points), 0.1 + 1e-9)
        # Geometric subdivision must keep the point count reasonable instead
        # of recursing all the way to the depth limit (2**16 + 1 points).
        self.assertLess(len(points), 1000)

    def test_s_shaped_quadratic(self) -> None:
        # A quadratic with control point far off the chord.
        curve = QuadraticBezier(0j, 50 - 100j, 100 + 0j)
        (points,) = Path(curve).flatten(0.5)
        self.assertGreater(len(points), 2)
        self.assertLessEqual(max_deviation(curve, points), 0.5 + 1e-9)

    def test_tolerance_controls_density(self) -> None:
        curve = CubicBezier(0j, 0 + 100j, 100 + 100j, 100 + 0j)
        fine = Path(curve).flatten(0.1)[0]
        coarse = Path(curve).flatten(5.0)[0]
        self.assertGreater(len(fine), len(coarse))
        self.assertLessEqual(max_deviation(curve, fine), 0.1 + 1e-9)
        self.assertLessEqual(max_deviation(curve, coarse), 5.0 + 1e-9)

    def test_large_arc_more_than_half_turn(self) -> None:
        # A large-arc sweep spanning well over 180 degrees. The midpoint can
        # sit near the chord while points close to an endpoint deviate a lot,
        # so the flattener must cap the angular span and subdivide.
        arc = Arc(
            complex(-64.06, 180.34),
            complex(160, 90),
            25,
            True,
            True,
            complex(65.29, 95.5),
        )
        tolerance = 0.1
        (points,) = Path(arc).flatten(tolerance)
        self.assertGreater(len(points), 4)
        self.assertLessEqual(max_deviation(arc, points), tolerance + 1e-9)

    def test_control_point_overshoots_chord_endpoint(self) -> None:
        # The control points project beyond the chord endpoints, so a
        # perpendicular-distance-to-infinite-line test wrongly accepts them.
        # The finite-segment distance test must subdivide this curve.
        curve = CubicBezier(
            complex(154.82, -64.70),
            complex(178.06, -83.78),
            complex(104.42, -27.22),
            complex(180.19, -76.13),
        )
        tolerance = 1.0
        (points,) = Path(curve).flatten(tolerance)
        self.assertLessEqual(max_deviation(curve, points), tolerance + 1e-9)
        self.assertGreater(len(points), 2)

    def test_degenerate_arcs_do_not_crash(self) -> None:
        # Coincident endpoints and zero-radius arcs are line/no-op cases
        # without a parameterized center or delta angle.
        noop = Arc(10 + 10j, 30 + 30j, 0, True, True, 10 + 10j)
        (points,) = Path(noop).flatten(0.1)
        self.assertEqual(points, [10 + 10j, 10 + 10j])

        linear = Arc(0j, 0 + 0j, 0, False, True, 20 + 0j)
        (points,) = Path(linear).flatten(0.1)
        self.assertEqual(points, [0j, 20 + 0j])

    def test_float_roundoff_keeps_closed_subpath_together(self) -> None:
        # Curves whose computed end differs from the subpath start by ~1e-14
        # must still be flattened into one closed subpath, not split in two.
        path = parse_path(
            "M 0,0 C 100,100 100,-100 200,0 A 50,30 20 1,1 380,40 L 400,40 Z"
        )
        subpaths = path.flatten(0.5)
        self.assertEqual(len(subpaths), 1)
        self.assertEqual(subpaths[0][0], subpaths[0][-1])


def max_deviation(segment: PathSegment, points: List[complex]) -> float:
    """Largest distance from the polyline to the sampled curve."""
    deviation = 0.0
    for i in range(1, 1000):
        p = segment.point(i / 1000)
        deviation = max(deviation, distance_to_polyline(p, points))
    return deviation


def distance_to_polyline(p: complex, points: List[complex]) -> float:
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
