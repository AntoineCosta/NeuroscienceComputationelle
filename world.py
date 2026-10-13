"""World: square arena with walls and static rectangular obstacles (top view, metres)."""
import numpy as np

from config import ARENA_SIZE

# (x_min, y_min, x_max, y_max). The robot starts at (0, 0) facing +y (up).
DEFAULT_OBSTACLES = [
    (-0.40, 1.00, 0.40, 1.25),     # straight ahead of the start: wall approach / back-off
    (0.75, -1.40, 1.05, -0.40),    # corridor (0.5 m wide) between these two blocks
    (1.55, -1.40, 1.85, -0.40),
]


class World:
    """
    Square arena (walls at +-size/2) with static axis-aligned rectangles.
    Obstacles only exist for the sensors and the display: the body passes through
    them (an overlap is shown, not prevented).
    """

    def __init__(self, obstacles=DEFAULT_OBSTACLES, size=ARENA_SIZE):
        self.size = size
        self.obstacles = np.array(obstacles, dtype=float).reshape(-1, 4)
        h = size / 2
        rects = np.vstack([[-h, -h, h, h], self.obstacles])
        # every rectangle edge as a segment (start point, end point)
        x0, y0, x1, y1 = rects.T
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        self.seg_start = np.vstack([np.column_stack(corners[k]) for k in range(4)])
        self.seg_end = np.vstack([np.column_stack(corners[(k + 1) % 4]) for k in range(4)])

    def ray_distance(self, origin, angle, max_range):
        """Distance from origin along angle to the nearest wall or obstacle edge (<= max_range)."""
        r = np.array([np.cos(angle), np.sin(angle)])
        s = self.seg_end - self.seg_start
        q = self.seg_start - origin
        denom = r[0] * s[:, 1] - r[1] * s[:, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (q[:, 0] * s[:, 1] - q[:, 1] * s[:, 0]) / denom     # along the ray
            u = (q[:, 0] * r[1] - q[:, 1] * r[0]) / denom           # along the segment
        hit = (denom != 0) & (t >= 0) & (u >= 0) & (u <= 1)
        return float(min(np.min(t[hit], initial=max_range), max_range))

    def overlaps(self, points):
        """True if any point is inside an obstacle or outside the arena."""
        points = np.atleast_2d(points)
        h = self.size / 2
        if np.any(np.abs(points) > h):
            return True
        x, y = points[:, :1], points[:, 1:]
        x0, y0, x1, y1 = self.obstacles.T
        return bool(np.any((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)))
