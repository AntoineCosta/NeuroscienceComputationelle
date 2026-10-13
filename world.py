"""Monde : arène carrée avec murs et obstacles rectangulaires fixes (vue de dessus, mètres)."""
import numpy as np

from config import ARENA_SIZE

# (x_min, y_min, x_max, y_max). Le robot part de (0, 0), orienté vers +y (vers le haut).
DEFAULT_OBSTACLES = [
    (-0.40, 1.00, 0.40, 1.25),     # droit devant le départ : approche du mur / recul
    (0.75, -1.40, 1.05, -0.40),    # couloir (0,5 m de large) entre ces deux blocs
    (1.55, -1.40, 1.85, -0.40),
]


class World:
    """
    Arène carrée (murs à +-size/2) avec des rectangles fixes alignés sur les axes.
    Les obstacles n'existent que pour les capteurs et l'affichage : le corps les
    traverse (le chevauchement est affiché, pas empêché).
    """

    def __init__(self, obstacles=DEFAULT_OBSTACLES, size=ARENA_SIZE):
        self.size = size
        self.obstacles = np.array(obstacles, dtype=float).reshape(-1, 4)
        h = size / 2
        rects = np.vstack([[-h, -h, h, h], self.obstacles])
        # chaque bord de rectangle sous forme de segment (point de départ, point d'arrivée)
        x0, y0, x1, y1 = rects.T
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        self.seg_start = np.vstack([np.column_stack(corners[k]) for k in range(4)])
        self.seg_end = np.vstack([np.column_stack(corners[(k + 1) % 4]) for k in range(4)])

    def ray_distance(self, origin, angle, max_range):
        """Distance depuis origin, selon angle, jusqu'au mur ou bord d'obstacle le plus proche (<= max_range)."""
        r = np.array([np.cos(angle), np.sin(angle)])
        s = self.seg_end - self.seg_start
        q = self.seg_start - origin
        denom = r[0] * s[:, 1] - r[1] * s[:, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (q[:, 0] * s[:, 1] - q[:, 1] * s[:, 0]) / denom     # position le long du rayon
            u = (q[:, 0] * r[1] - q[:, 1] * r[0]) / denom           # position le long du segment
        hit = (denom != 0) & (t >= 0) & (u >= 0) & (u <= 1)
        return float(min(np.min(t[hit], initial=max_range), max_range))

    def overlaps(self, points):
        """True si un des points est dans un obstacle ou hors de l'arène."""
        points = np.atleast_2d(points)
        h = self.size / 2
        if np.any(np.abs(points) > h):
            return True
        x, y = points[:, :1], points[:, 1:]
        x0, y0, x1, y1 = self.obstacles.T
        return bool(np.any((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)))
