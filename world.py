"""Static map model: height grid, line of sight, navigation graph."""
import math

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

G = 4  # nav grid step in meters


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 1


class World:
    def __init__(self, path, cfg):
        d = np.load(path)
        self.S = d["surface"].astype(np.float32) / 100.0
        self.ny, self.nx = self.S.shape
        self.minx = float(d["minx"]) if "minx" in d else -513.0
        self.miny = float(d["miny"]) if "miny" in d else -334.0
        mech = cfg["mech"]
        self.step = float(mech["step_up"])
        self.walk = float(mech["walk_speed"])
        self.max_climb = 38.0
        S = self.S
        # clearance: no wall inside the mech radius
        self.free = (maximum_filter(S, footprint=_disk(2)) - S) <= 1.6
        self.free[:4, :] = False
        self.free[-4:, :] = False
        self.free[:, :4] = False
        self.free[:, -4:] = False
        self._build_graph()
        self._cache = {}

    # ---------------------------------------------------------------- grid helpers
    def cell(self, x, y):
        i = int(round(x - self.minx))
        j = int(round(y - self.miny))
        return min(max(j, 0), self.ny - 1), min(max(i, 0), self.nx - 1)

    def height(self, x, y):
        j, i = self.cell(x, y)
        return float(self.S[j, i])

    def heights(self, xs, ys):
        i = np.clip(np.rint(xs - self.minx).astype(np.int32), 0, self.nx - 1)
        j = np.clip(np.rint(ys - self.miny).astype(np.int32), 0, self.ny - 1)
        return self.S[j, i]

    def is_free(self, xs, ys):
        i = np.clip(np.rint(xs - self.minx).astype(np.int32), 0, self.nx - 1)
        j = np.clip(np.rint(ys - self.miny).astype(np.int32), 0, self.ny - 1)
        return self.free[j, i]

    def los(self, a, b):
        """a, b: arrays (..., 3). True where the segment clears the surface."""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        a, b = np.broadcast_arrays(a, b)
        dmax = float(np.max(np.hypot(b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]))) if a.size else 0.0
        k = int(min(max(dmax / 1.5, 4), 160))
        t = np.linspace(0.04, 0.96, k, dtype=np.float32)
        xs = a[..., 0:1] + (b[..., 0:1] - a[..., 0:1]) * t
        ys = a[..., 1:2] + (b[..., 1:2] - a[..., 1:2]) * t
        zs = a[..., 2:3] + (b[..., 2:3] - a[..., 2:3]) * t
        return np.all(self.heights(xs, ys) < zs - 0.15, axis=-1)

    def los1(self, a, b):
        return bool(self.los(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)))

    def walkable_line(self, x0, y0, x1, y1, max_drop=1.6):
        d = math.hypot(x1 - x0, y1 - y0)
        k = int(d) + 2
        t = np.linspace(0.0, 1.0, k)
        xs = x0 + (x1 - x0) * t
        ys = y0 + (y1 - y0) * t
        h = self.heights(xs, ys)
        dh = np.diff(h)
        return bool(np.all(self.is_free(xs[1:], ys[1:])) and np.all(dh <= self.step + 0.2) and np.all(dh >= -max_drop))

    def walkable_to(self, x0, y0, xs, ys, k=16, max_drop=1.6):
        """Vectorized straight-walk test from one point to many."""
        t = np.linspace(0.0, 1.0, k, dtype=np.float32)
        px = x0 + (xs[:, None] - x0) * t
        py = y0 + (ys[:, None] - y0) * t
        h = self.heights(px, py)
        dh = np.diff(h, axis=1)
        seg = np.hypot(xs - x0, ys - y0)[:, None] / (k - 1)
        up_ok = np.all(dh <= np.maximum(1.0, seg) * (self.step + 0.2), axis=1)
        dn_ok = np.all(dh >= -max_drop * np.maximum(1.0, seg), axis=1)
        return up_ok & dn_ok & np.all(self.is_free(px[:, 1:], py[:, 1:]), axis=1)

    # ---------------------------------------------------------------- nav graph
    def _build_graph(self):
        S, free = self.S, self.free
        self.gy = (self.ny - 2) // G
        self.gx = (self.nx - 2) // G
        cj = 1 + G * np.arange(self.gy)
        ci = 1 + G * np.arange(self.gx)
        self.nh = S[np.ix_(cj, ci)]
        self.nfree = free[np.ix_(cj, ci)]
        JJ, II = np.meshgrid(np.arange(self.gy), np.arange(self.gx), indexing="ij")
        rows, cols, wts = [], [], []
        R = 2
        for dj in range(-R, R + 1):
            for di in range(-R, R + 1):
                if dj == 0 and di == 0:
                    continue
                j0 = max(0, -dj)
                j1 = self.gy - max(0, dj)
                i0 = max(0, -di)
                i1 = self.gx - max(0, di)
                aj = JJ[j0:j1, i0:i1]
                ai = II[j0:j1, i0:i1]
                bj = aj + dj
                bi = ai + di
                ok = self.nfree[aj, ai] & self.nfree[bj, bi]
                ha = self.nh[aj, ai]
                hb = self.nh[bj, bi]
                dz = hb - ha
                L = G * math.hypot(dj, di)
                n = int(round(L))
                # 1 m profile along the line
                prof = []
                clear = np.ones_like(ok)
                for s in range(n + 1):
                    f = s / n
                    pj = np.rint(1 + G * aj + f * G * dj).astype(np.int32)
                    pi = np.rint(1 + G * ai + f * G * di).astype(np.int32)
                    prof.append(S[pj, pi])
                    clear &= free[pj, pi]
                prof = np.stack(prof, axis=0)
                rise = np.diff(prof, axis=0)
                unit = L / n
                adjacent = max(abs(dj), abs(di)) == 1
                if adjacent:
                    walk = ok & clear & np.all(rise <= (self.step + 0.05) * max(1.0, unit), axis=0)
                    drop = np.maximum(0.0, -dz)
                    w = L / self.walk + np.where(drop > 3.0, 1.0 + drop * 0.05, 0.0)
                    self._add(rows, cols, wts, walk, aj, ai, bj, bi, w)
                else:
                    walk = np.zeros_like(ok)
                mid_max = prof[1:-1].max(axis=0) if n > 1 else np.minimum(ha, hb)
                # jet up onto a ledge or roof
                jet = ok & ~walk & (dz > self.step) & (dz <= self.max_climb) & (mid_max <= hb + 0.3)
                w = L / 8.0 + dz / 8.0 + dz * 0.12 + 3.0
                self._add(rows, cols, wts, jet, aj, ai, bj, bi, w)
                # step off an edge
                if not adjacent:
                    fall = ok & (dz < -self.step) & (mid_max <= ha + 0.3)
                    w = L / self.walk + 1.0 + (-dz) * 0.05
                    self._add(rows, cols, wts, fall, aj, ai, bj, bi, w)
        n_nodes = self.gy * self.gx
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        w = np.concatenate(wts).astype(np.float32)
        # transposed: search runs from the goal
        self.GT = coo_matrix((w.astype(np.float64), (c, r)), shape=(n_nodes, n_nodes)).tocsr()
        self.budget = 1

    def _add(self, rows, cols, wts, mask, aj, ai, bj, bi, w):
        if not mask.any():
            return
        a = (aj * self.gx + ai)[mask]
        b = (bj * self.gx + bi)[mask]
        rows.append(a)
        cols.append(b)
        wts.append(np.broadcast_to(w, mask.shape)[mask])

    def penalize(self, n, cost):
        """Make every edge into node n cost more, after the bot got stuck heading there."""
        a, b = self.GT.indptr[n], self.GT.indptr[n + 1]
        self.GT.data[a:b] += cost
        self._cache.clear()

    def node(self, x, y):
        gi = int(round((x - self.minx - 1) / G))
        gj = int(round((y - self.miny - 1) / G))
        return min(max(gj, 0), self.gy - 1), min(max(gi, 0), self.gx - 1)

    def node_xy(self, n):
        gj, gi = divmod(int(n), self.gx)
        return self.minx + 1 + G * gi, self.miny + 1 + G * gj, float(self.nh[gj, gi])

    def free_node(self, x, y, z=None, r=4):
        """Nearest free node id, preferring nodes near height z."""
        gj, gi = self.node(x, y)
        best, bd = None, 1e9
        for dj in range(-r, r + 1):
            j = gj + dj
            if j < 0 or j >= self.gy:
                continue
            for di in range(-r, r + 1):
                i = gi + di
                if i < 0 or i >= self.gx or not self.nfree[j, i]:
                    continue
                d = G * math.hypot(dj, di)
                if z is not None:
                    d += 3.0 * abs(float(self.nh[j, i]) - z)
                if d < bd:
                    bd, best = d, j * self.gx + i
        return best

    def field(self, goal, limit=np.inf):
        """(dist, next_hop) toward goal node. Cached. None when the tick budget is spent."""
        gj, gi = divmod(goal, self.gx)
        for k, f in self._cache.items():
            kj, ki = divmod(k, self.gx)
            if abs(kj - gj) <= 1 and abs(ki - gi) <= 1 and f[2] >= limit:
                return f
        if self.budget <= 0:
            return None
        self.budget -= 1
        dist, pred = dijkstra(self.GT, directed=True, indices=goal, return_predecessors=True, limit=limit)
        self._cache.pop(goal, None)
        if len(self._cache) > 6:
            self._cache.pop(next(iter(self._cache)))
        f = (dist, pred, limit)
        self._cache[goal] = f
        return f
