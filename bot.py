"""Mech Royale bot. Reads mechfight/1 JSON lines on stdin, answers on stdout."""
import json
import math
import os
import queue
import random
import sys
import threading
import time
import traceback

import numpy as np

from world import World

HERE = os.path.dirname(os.path.abspath(__file__))
NOCOVER = bool(os.environ.get("MECH_NOCOVER"))  # A/B switch for local runs
PREF = {"laser": 3, "lightning": 2, "mortar": 1, "gun": 0}
UPGRADES = ["plating", "armor", "nanites", "gun_mag", "gun_damage", "laser_damage", "gun_rate", "laser_charge",
            "servos", "gun_velocity", "gyros", "lightning_chain", "lightning_range",
            "mortar_reload", "capacitor", "reactor", "jets", "sensors"]
# the same list ordered by fighting power: a bigger magazine is +31% sustained gun damage, armor only 9% more life
UPGRADES2 = ["gun_mag", "gun_damage", "plating", "gun_rate", "laser_damage", "armor", "laser_charge", "gun_velocity",
             "nanites", "servos", "gyros", "lightning_efficiency", "lightning_chain", "lightning_range",
             "mortar_reload", "capacitor", "reactor", "jets", "sensors"]
# tunables, overridden per run for local A/B tests
P = {"retreat_hp": 0.0, "recover_hp": 0.8, "range": 75.0, "peek_max": 9.0, "close": 30.0,
     "danger_mem": 0.3,   # s to stay hidden after seeing a laser charging at us
     "crowd": 99,         # hide instead of trading when this many enemies were seen in the last 2 s
     "crowd_r": 120.0,
     "los_bias": 50.0,     # m added to the target score of an enemy we cannot hit right now
     "far_always": 0,     # 1: search far cover whenever no near cover exists
     "mem_s": 1.0,        # s a mech stays a threat after the server stops listing it (the server already tracks 5 s)
     "kills": 1,          # 1: forget a mech the moment the server reports our kill
     "laser_disc": 1,     # 1: start a laser charge only when a shot is likely before the hold timer runs out
     "peek_charge": 6.0,  # m to the peek spot under which a charge behind cover is allowed
     "hear": 1,           # 1: with no threat known, face the last near sound
     "glance": 1,         # 1: turn to an attacker we cannot see
     "laser_hide": 0,     # 1: break line of sight for the whole time an enemy laser can still fire at us
     "laser_hold": 2.8,   # s a charge can be kept before the server fires it (0.75 charge + 2 hold)
     "grab": 45.0,        # m: with an empty second slot, leave the fight line to take a weapon this close
     "ltng_gap": 1,       # 1: back away from a lightning holder inside 65 m even when cover exists
     "shell_dodge": 1,    # 1: run from the landing point of a mortar shell
     "cube_r": 120.0,     # m: in a fight, walk onto power cubes this close (+4% HP and damage each, upgrades)
     "drop_r": 300.0,     # m: in a fight, go and stand in a supply drop ring this close (full HP, 3 cubes)
     "retreat2_hp": 0.0,  # HP share under which we break off when two or more enemies are in contact
     "finish": 0.0,       # no retreat while the target has less than this share of our own HP
     "quiet": 0,          # 1: in retreat, shoot only a close or nearly dead target, so regen can start
     "heal_stay": 0,      # 1: in retreat with nobody in sight, wait at the cover spot instead of looting
     "laser_range": 0.0,  # m: open-ground fighting distance while we hold a laser, 0 keeps "range"
     "ups": 0,            # 1: pick upgrades by fighting power (UPGRADES2)
     "picky": 0,          # 1: fight only mechs that hurt us, aim at us or stand close; ignore the rest
     "picky_r": 40.0,     # m: with picky, a mech this close always counts
     "picky_aim": 0.2}    # rad: with picky, a mech whose aim is this near to us for 0.3 s counts
P.update({'ups': 1})
try:
    P.update(json.loads(os.environ.get("MECH_P") or "{}"))
except Exception:
    pass


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def adiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def ring(rmax, step):
    pts = []
    n = int(rmax / step)
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            d = math.hypot(i, j) * step
            if 0 < d <= rmax:
                pts.append((i * step, j * step))
    a = np.array(pts, dtype=np.float32)
    return a[:, 0], a[:, 1]


class Bot:
    def __init__(self, hello):
        self.cfg = hello["config"]
        self.W = World(os.path.join(HERE, "data", "map.npz"), self.cfg)
        self.rng = random.Random(hello.get("seed", 1))
        m = self.cfg["mech"]
        self.size_per_cube = self.cfg["cubes"].get("size_per_cube", 0.05)
        self.h_mech = m["height"]
        self.known = {}
        self.hostile = set()
        self.aim_n = {}
        self.items = {}
        self.retreat = False
        self.last_dmg_t = -99.0
        self.dmg_by = None
        self.snd_t = -99.0
        self.snd_bearing = 0.0
        self.charge_ok = False
        self.glance_until = -9.0
        self.lz = {}
        self.dmg_bearing = 0.0
        self.drops_now = []
        self.last_threat_t = -99.0
        self.strafe_sign = 1
        self.strafe_until = 0.0
        self.stuck = 0
        self.unstick_until = 0.0
        self.unstick_dir = (0.0, 0.0)
        self.jet = None
        self.P = dict(P)
        self.nav_used = False
        self.nav_hop = None
        self.no_short_until = 0.0
        self.stuck_at = None
        self.stuck_n = 0
        self.spot = None
        self.spot_t = -9.0
        self.bad_spots = []
        self.cover = None
        self.cover_t = -9.0
        self.near = ring(26, 2.0)
        self.far = ring(78, 6.0)
        self.aim = [0.0, 0.0]
        self.blind_since = None
        try:
            with open(os.path.join(HERE, "data", "positions.json")) as f:
                self.forts = json.load(f)
        except Exception:
            self.forts = []

    # ------------------------------------------------------------------ helpers
    def scale(self, cubes):
        return 1.0 + self.size_per_cube * (cubes or 0)

    def tgt(self, m):
        """Aim point (center) and radius of a mech, npc or crate record."""
        k = m.get("kind")
        p = m["pos"]
        if k == "npc":
            return [p[0], p[1], p[2] + 1.75], 1.8
        if k == "crate":
            return [p[0], p[1], p[2] + 1.2], 1.4
        sc = self.scale(m.get("cubes"))
        return [p[0], p[1], p[2] + 4.5 * sc], 2.5 * sc

    def muzzle(self):
        p = self.me["pos"]
        return [p[0], p[1], p[2] + self.mz]

    def can_hit(self, m):
        c, _ = self.tgt(m)
        return self.W.los1(self.muzzle(), c)

    def pick(self, offer):
        best, bs = 0, 1e9
        for i, c in enumerate(offer.get("choices", [])):
            cid = c.get("id", "")
            order = UPGRADES2 if self.P["ups"] else UPGRADES
            s = order.index(cid) if cid in order else 50
            if s < bs:
                best, bs = i, s
        return best

    # ------------------------------------------------------------------ memory
    def remember(self, o):
        t = self.t
        p = self.me["pos"]
        for m in o.get("mechs", []):
            r = dict(m)
            r["seen"] = t
            self.known[m["id"]] = r
            if m["kind"] == "mech" and m["id"] not in self.hostile and m.get("aim"):
                q = m["pos"]
                d = math.hypot(p[0] - q[0], p[1] - q[1])
                off = abs((math.atan2(p[1] - q[1], p[0] - q[0]) - m["aim"][0] + math.pi) % (2 * math.pi) - math.pi)
                if d < self.P["picky_r"]:
                    self.hostile.add(m["id"])
                elif off < self.P["picky_aim"] and d < 260.0:
                    n = self.aim_n.get(m["id"], 0) + 1
                    self.aim_n[m["id"]] = n
                    if n >= 3:
                        self.hostile.add(m["id"])
                else:
                    self.aim_n[m["id"]] = 0
        for k in list(self.known):
            r = self.known[k]
            if t - r["seen"] > (6.0 if r["kind"] == "mech" else 25.0):
                del self.known[k]
        vis = set()
        for typ in ("crates", "cubes", "pickups"):
            for it in o.get(typ, []):
                vis.add(it["id"])
                self.items[it["id"]] = {"id": it["id"], "type": typ[:-1], "pos": it["pos"], "kind": it.get("kind"), "seen": t}
        eye = [p[0], p[1], p[2] + 8.0]
        for k in list(self.items):
            if k in vis:
                continue
            q = self.items[k]["pos"]
            if math.hypot(q[0] - p[0], q[1] - p[1]) < 38 and self.W.los1(eye, [q[0], q[1], q[2] + 1.0]):
                del self.items[k]
        for sd in o.get("sounds", []):
            if sd.get("dist") == "near" and sd.get("kind") in ("step", "boost", "gun", "laser_charge", "laser", "reload", "jet", "lightning"):
                self.snd_t = t - (sd.get("age") or 0.0)
                self.snd_bearing = sd["bearing"]
        for e in o.get("events", []):
            if e["type"] == "damaged" and e.get("bearing") is not None:
                self.last_dmg_t = t
                self.dmg_bearing = e["bearing"]
                self.dmg_by = e.get("by")
                if e.get("by") is not None:
                    self.hostile.add(e["by"])
            elif e["type"] == "kill" and self.P["kills"]:
                r = self.known.pop(e.get("target"), None)
                if r is not None:
                    # the dead mech left its cubes where it stood, also when we cannot see the spot
                    self.items["k%s" % r["id"]] = {"id": "k%s" % r["id"], "type": "cube", "pos": list(r["pos"]), "kind": None, "seen": t}
            elif e["type"] == "took":
                self.items.pop(e.get("id"), None)

    # ------------------------------------------------------------------ movement
    def direct(self, a, x, y, stop=0.6):
        p = self.me["pos"]
        dx, dy = x - p[0], y - p[1]
        d = math.hypot(dx, dy)
        if d < stop:
            a["move"] = [0.0, 0.0]
        else:
            k = min(1.0, d / 2.0) / d
            a["move"] = [dx * k, dy * k]
        return d

    def nav(self, a, x, y, z=None, boost=False):
        """Path-follow toward (x, y). Returns remaining straight distance."""
        W, me = self.W, self.me
        p = me["pos"]
        dgoal = math.hypot(x - p[0], y - p[1])
        self.nav_used = True
        if self.jet is not None:
            j = self.jet
            self.direct(a, j["x"], j["y"], stop=0.3)
            a["boost"] = p[2] < j["h"] + 1.0 and me["energy"] > 1
            return dgoal
        gn = W.free_node(x, y, z, r=5)
        mn = W.free_node(p[0], p[1], p[2], r=3)
        if gn is None or mn is None:
            self.direct(a, x, y)
            return dgoal
        f = W.field(gn, dgoal / W.walk * 2.5 + 15.0)
        if f is not None and not np.isfinite(f[0][mn]) and np.isfinite(f[2]):
            f = W.field(gn)
        if f is None or not np.isfinite(f[0][mn]):
            self.direct(a, x, y)
            return dgoal
        dist, pred = f[0], f[1]
        if mn == gn or dgoal < 3.0:
            self.direct(a, x, y)
            return dgoal
        path = []
        n = mn
        for _ in range(6):
            n = int(pred[n])
            if n < 0:
                break
            path.append(n)
        if not path:
            self.direct(a, x, y)
            return dgoal
        target = None
        self.nav_hop = path[0]
        if self.t >= self.no_short_until:
            for n in reversed(path):
                tx, ty, th = W.node_xy(n)
                if abs(th - p[2]) < 1.5 and W.walkable_line(p[0], p[1], tx, ty):
                    target = (tx, ty, th)
                    break
        if target is None:
            target = W.node_xy(path[0])
            tx, ty, th = target
            # off the node center a checked edge can clip a corner: line up on the node first
            if abs(th - p[2]) < 1.5 and not W.walkable_line(p[0], p[1], tx, ty):
                cx, cy, ch = W.node_xy(mn)
                if math.hypot(cx - p[0], cy - p[1]) > 1.0 and abs(ch - p[2]) < 1.5 and W.walkable_line(p[0], p[1], cx, cy):
                    target = (cx, cy, ch)
        tx, ty, th = target
        if th > p[2] + 1.3 and me["grounded"]:
            need = min(me["max_energy"] - 3.0, (th - p[2]) * 1.7 + 12.0)
            if me["energy"] < need:
                a["move"] = [0.0, 0.0]
                return dgoal
            a["jump"] = True
            a["boost"] = True
            self.jet = {"x": tx, "y": ty, "h": th, "t0": self.t, "node": path[0]}
            self.direct(a, tx, ty, stop=0.3)
            return dgoal
        dx, dy = tx - p[0], ty - p[1]
        d = math.hypot(dx, dy) or 1.0
        a["move"] = [dx / d, dy / d]
        if boost and me["energy"] > 25:
            a["boost"] = True
        return dgoal

    def unstick(self, a):
        me = self.me
        p = me["pos"]
        mv = a.get("move", [0, 0])
        sp = math.hypot(me["vel"][0], me["vel"][1])
        if self.t < self.unstick_until:
            a["move"] = list(self.unstick_dir)
            a["boost"] = me["energy"] > 5
            return
        if math.hypot(mv[0], mv[1]) > 0.5 and sp < 1.5 and me["grounded"] and self.jet is None:
            self.stuck += 1
        else:
            self.stuck = 0
        if self.stuck < 6:
            return
        self.stuck = 0
        sa = self.stuck_at
        if sa and math.hypot(p[0] - sa[0], p[1] - sa[1]) < 8.0 and self.t - sa[2] < 12.0:
            self.stuck_n += 1
        else:
            self.stuck_n = 1
        self.stuck_at = (p[0], p[1], self.t)
        n = self.stuck_n
        # whatever we were heading for is not reachable this way: make it cost more
        if self.nav_used and self.nav_hop is not None:
            self.W.penalize(self.nav_hop, 8.0 * n)
            self.no_short_until = self.t + 6.0
        if self.spot is not None and self.spot_t == self.t:
            self.bad_spots.append((self.spot[0], self.spot[1], self.t + 20.0))
            self.bad_spots = self.bad_spots[-8:]
            self.cover = None
        self.strafe_sign = -self.strafe_sign
        self.strafe_until = self.t + 1.0
        s = self.rng.choice((-1, 1))
        back = 0.3 if n == 1 else -0.7
        self.unstick_dir = (-mv[1] * s * 0.8 + mv[0] * back, mv[0] * s * 0.8 + mv[1] * back)
        self.unstick_until = self.t + (0.7 if n == 1 else 1.2)
        a["move"] = list(self.unstick_dir)
        a["jump"] = True
        a["boost"] = True

    # ------------------------------------------------------------------ cover
    def find_cover(self, threats, primary, offs, straight):
        W, me = self.W, self.me
        p = me["pos"]
        xs = p[0] + offs[0]
        ys = p[1] + offs[1]
        h = W.heights(xs, ys)
        ok = W.is_free(xs, ys)
        zc = self.zone["center"]
        ok &= np.hypot(xs - zc[0], ys - zc[1]) < self.zone["radius"] - 6
        for bx, by, until in self.bad_spots:
            if until > self.t:
                ok &= np.hypot(xs - bx, ys - by) > 3.0
        if straight:
            ok &= np.abs(h - p[2]) < 2.5
            ok &= W.walkable_to(p[0], p[1], xs, ys)
        else:
            ok &= np.abs(h - p[2]) < 6.0
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            return None
        cand = np.stack([xs[idx], ys[idx], h[idx]], axis=1)
        sc = self.scale(me["cubes"])
        body = cand + np.array([0, 0, 4.5 * sc], dtype=np.float32)
        head = cand + np.array([0, 0, 8.5 * sc], dtype=np.float32)
        gunp = cand + np.array([0, 0, self.mz], dtype=np.float32)
        exposed = np.zeros(len(idx), dtype=np.int32)
        can_fire = np.zeros(len(idx), dtype=bool)
        for th in threats[:4]:
            q = th["pos"]
            ts = self.scale(th.get("cubes"))
            mz = np.array([q[0], q[1], q[2] + 7.0 * ts], dtype=np.float32)
            vis = W.los(mz, body) | W.los(mz, head)
            exposed += vis
            if th is primary:
                c, _ = self.tgt(th)
                can_fire = W.los(gunp, np.array(c, dtype=np.float32))
        d = np.hypot(cand[:, 0] - p[0], cand[:, 1] - p[1])
        hidden = exposed == 0
        if not hidden.any():
            return None
        hid = np.nonzero(hidden)[0]
        hid = hid[np.argsort(d[hid])][:14]
        pk = np.nonzero(can_fire & (exposed <= 1))[0]
        best = None
        for hi in hid:
            if len(pk):
                dp = np.hypot(cand[pk, 0] - cand[hi, 0], cand[pk, 1] - cand[hi, 1])
                k = int(np.argmin(dp))
                dpk = float(dp[k])
                peek = cand[pk[k]] if dpk <= self.P["peek_max"] else None
            else:
                dpk, peek = 12.0, None
            s = float(d[hi]) + 1.5 * min(dpk, 12.0)
            if best is None or s < best[0]:
                best = (s, cand[hi], peek)
        return {"hide": [float(v) for v in best[1]], "peek": None if best[2] is None else [float(v) for v in best[2]],
                "straight": straight}

    # ------------------------------------------------------------------ weapons
    def shoot(self, a, m, allow=True):
        """Aim at record m and pull whatever trigger makes sense. Returns True if aligned."""
        me = self.me
        p = me["pos"]
        mu = self.muzzle()
        c, rad = self.tgt(m)
        v = m.get("vel") or [0, 0, 0]
        age = self.t - m.get("seen", self.t)
        c = [c[0] + v[0] * age, c[1] + v[1] * age, c[2]]
        d = math.dist(mu, c)
        fire = [False, False]
        ws = me["weapons"]
        los = self.W.los1(mu, c)
        visible = age < 0.05
        aim_pt = c
        charged = False
        for i, w in enumerate(ws):
            if w and w["kind"] == "laser":
                full = w["stats"].get("charge", 0.75)
                ch = w.get("charge", 0) or 0
                if ch >= 0.55 * full:
                    charged = True
        gun = next((w for w in ws if w and w["kind"] == "gun"), None)
        if gun and not charged:
            sp = gun["stats"]["speed"]
            tt = d / sp
            for _ in range(3):
                q = [c[0] + v[0] * (tt + 0.08), c[1] + v[1] * (tt + 0.08), c[2]]
                tt = math.dist(mu, q) / sp
            aim_pt = q
        dx, dy, dz = aim_pt[0] - mu[0], aim_pt[1] - mu[1], aim_pt[2] - mu[2]
        hd = math.hypot(dx, dy)
        yaw, pitch = math.atan2(dy, dx), math.atan2(dz, hd)
        gyaw, gpitch = yaw, pitch
        # mortar takes over the aim when it is ready and the shot is sane
        for i, w in enumerate(ws):
            if w and w["kind"] == "mortar" and m.get("kind") == "mech" and w["cooldown"] <= 0.25 and allow and visible:
                sol = self.mortar(w, m)
                if sol is not None:
                    yaw, pitch = sol
                    if abs(adiff(me["aim"][0], yaw)) < 0.03 and abs(me["aim"][1] - pitch) < 0.03:
                        fire[i] = True
        a["aim"] = [yaw, pitch]
        # direct-fire weapons are judged on the direct line, also while the mortar arc owns the aim
        ey = abs(adiff(me["aim"][0], gyaw))
        ep = abs(me["aim"][1] - gpitch)
        tol = math.atan2(rad * 0.8, max(d, 1.0)) + 0.008
        aligned = ey < tol and ep < tol + 0.01
        for i, w in enumerate(ws):
            if not w:
                continue
            k = w["kind"]
            if k == "gun":
                if allow and aligned and los and d <= w["stats"]["range"] and w.get("ammo", 1) > 0:
                    fire[i] = True
            elif k == "laser":
                ch = w.get("charge", 0) or 0
                full = w["stats"].get("charge", 0.75)
                rng = w["stats"].get("range", 280)
                if ch > 0:
                    # hold until the beam will land, the server fires it at hold_s anyway
                    ready = ch >= 0.95 * full
                    fire[i] = not (ready and aligned and los and visible)
                    if ch >= full + 1.6:
                        fire[i] = False
                elif w["cooldown"] <= 0.05 and m.get("kind") == "mech" and d < rng and allow:
                    # a charge that finds no line within the hold time is a wasted beam at half walking speed
                    if not self.P["laser_disc"] or (los and visible) or self.charge_ok:
                        fire[i] = True
            elif k == "lightning":
                if allow and los and d <= w["stats"]["range"] * 0.97 and ey < 0.2 and me["energy"] > 12:
                    fire[i] = True
        a["fire"] = fire
        return aligned and los

    def mortar(self, w, m):
        p = self.me["pos"]
        v = w["stats"]["speed"]
        g = self.cfg["weapons"]["mortar"]["gravity"]
        q = m["pos"]
        tv = m.get("vel") or [0, 0, 0]
        if math.hypot(tv[0], tv[1]) > 9:
            return None
        sc = self.scale(self.me["cubes"])
        oz = p[2] + self.h_mech * sc + 0.5
        tx, ty = q[0], q[1]
        for _ in range(2):
            x = math.hypot(tx - p[0], ty - p[1])
            if x < 20 or x > 170:
                return None
            y = q[2] - oz
            disc = v ** 4 - g * (g * x * x + 2 * y * v * v)
            if disc < 0:
                return None
            th = math.atan2(v * v - math.sqrt(disc), g * x)
            tf = x / (v * math.cos(th))
            tx, ty = q[0] + tv[0] * tf, q[1] + tv[1] * tf
        if th < self.cfg["mech"]["min_pitch"] or th > self.cfg["mech"]["max_pitch"]:
            return None
        return math.atan2(ty - p[1], tx - p[0]), th

    def take_loot(self, a, o):
        me = self.me
        p = me["pos"]
        ws = me["weapons"]
        held = {w["kind"] for w in ws if w}
        for it in o.get("pickups", []):
            q = it["pos"]
            if math.hypot(q[0] - p[0], q[1] - p[1]) > 3.7 or abs(q[2] - p[2]) > 3.8 or it["kind"] in held:
                continue
            if ws[1] is None:
                a["take"] = it["id"]
                return
            if PREF.get(it["kind"], 0) > PREF.get(ws[1]["kind"], 0) and not (ws[1].get("charge") or 0) > 0:
                a["take"] = it["id"]
                a["drop"] = 1
                return

    # ------------------------------------------------------------------ decisions
    def act(self, o):
        me = self.me = o["self"]
        t = self.t = o["t"]
        p = me["pos"]
        st = me.get("stats") or {}
        self.mz = st.get("muzzle_height", self.cfg["mech"]["muzzle_height"] * self.scale(me["cubes"]))
        z = self.zone = o["zone"]
        a = {"move": [0.0, 0.0], "fire": [False, False]}
        if o.get("offer"):
            a["pick"] = self.pick(o["offer"])
        self.W.budget = 1
        self.nav_used = False
        if self.jet is not None:
            j = self.jet
            if t - j["t0"] > 6.0 or (me["grounded"] and t - j["t0"] > 0.4):
                if me["grounded"] and p[2] < j["h"] - 1.0:
                    self.W.penalize(j["node"], 15.0)  # the jump did not get us up there
                self.jet = None
        self.remember(o)
        self.take_loot(a, o)
        self.drops_now = o.get("drops") or []

        nc, nr = z["next_center"], z["next_radius"]
        dn = math.hypot(p[0] - nc[0], p[1] - nc[1])
        dc = math.hypot(p[0] - z["center"][0], p[1] - z["center"][1])
        margin = min(25.0, nr * 0.4)
        out = dn - (nr - margin)
        must_go = out > 0 and t + out / 7.5 + 6.0 >= z["shrink_start"]
        outside = dc > z["radius"] - 4.0
        self.must_go = must_go

        threats = [m for m in self.known.values() if m["kind"] == "mech" and t - m["seen"] <= self.P["mem_s"]]
        if self.P["picky"]:
            threats = [m for m in threats if m["id"] in self.hostile]
        primary = None
        self.charge_ok = False
        if threats:
            def score(m):
                d = math.dist(m["pos"], p)
                return d * (0.5 + m.get("hp", 1400) / max(1.0, m.get("max_hp", 1400))) + 40.0 * (t - m["seen"])
            threats.sort(key=score)
            lb = self.P["los_bias"]
            if lb > 0 and len(threats) > 1:
                # shoot back at what can be hit, not at the weakest name on the list
                top = threats[:4]
                sc = [score(m) + (0.0 if (t - m["seen"] < 0.3 and self.can_hit(m)) else lb) for m in top]
                order = sorted(range(len(top)), key=lambda i: sc[i])
                threats[:4] = [top[i] for i in order]
            primary = threats[0]
            self.last_threat_t = t
        hpf = me["hp"] / max(1.0, me["max_hp"])
        rh = self.P["retreat_hp"]
        if self.P["retreat2_hp"] > rh and sum(1 for m in threats if math.dist(m["pos"], p) < 160.0) >= 2:
            rh = self.P["retreat2_hp"]
        # a target far weaker than us is finished first: its death is a point and often ends the fight
        fin = primary is not None and primary.get("hp", 1e9) < self.P["finish"] * me["hp"]
        if fin:
            self.retreat = False
        elif hpf < rh:
            self.retreat = True
        elif hpf > self.P["recover_hp"] or (self.retreat and not self.P["heal_stay"] and t - self.last_threat_t > 7.0 and hpf > 0.6):
            self.retreat = False

        if outside or (must_go and (primary is None or out > 35.0 or t >= z["shrink_start"] - 2.0)):
            self.mode = "zone"
            self.go_zone(a, nc, nr, margin, dn, outside or t >= z["shrink_start"] - 4.0)
            if primary is not None:
                self.shoot(a, primary)
            else:
                self.look(a)
        elif primary is not None:
            self.mode = "fight"
            self.fight(a, primary, threats)
        elif self.retreat and self.P["heal_stay"] and self.spot is not None and t - self.spot_t < 40.0:
            # hurt and out of sight: regen gives 45 HP a second after 5 quiet seconds, so wait behind the cover
            self.mode = "heal"
            s = self.spot
            if math.hypot(s[0] - p[0], s[1] - p[1]) > 2.0:
                self.nav(a, s[0], s[1], s[2])
                if math.hypot(s[0] - p[0], s[1] - p[1]) < 6.0:
                    self.direct(a, s[0], s[1], stop=0.5)
            self.look(a)
        else:
            self.mode = "loot"
            self.loot(a, o, nc, nr, dn)
        if self.jet is not None and not self.nav_used:
            # a jump in progress is finished whatever the mode, half a jump only falls back
            j = self.jet
            self.direct(a, j["x"], j["y"], stop=0.3)
            a["boost"] = p[2] < j["h"] + 1.0 and me["energy"] > 1
        self.unstick(a)
        if self.P["shell_dodge"]:
            g = self.cfg["weapons"]["mortar"]["gravity"]
            for pr in o.get("projectiles", []):
                if pr.get("kind") != "mortar" or pr.get("by") == me.get("id"):
                    continue
                q, v = pr["pos"], pr.get("vel") or [0.0, 0.0, 0.0]
                tf = (v[2] + math.sqrt(v[2] * v[2] + 2.0 * g * max(0.0, q[2] - p[2]))) / g
                dx, dy = p[0] - (q[0] + v[0] * tf), p[1] - (q[1] + v[1] * tf)
                dd = math.hypot(dx, dy)
                if dd < 10.0 and tf < 2.5:
                    if dd < 0.5:
                        dx, dy = -v[1], v[0]
                        dd = math.hypot(dx, dy) or 1.0
                    a["move"] = [dx / dd, dy / dd]
                    a["boost"] = me["energy"] > 5
                    break
        if self.P["glance"] and self.dmg_by is not None:
            if t - self.last_dmg_t < 0.15 and not any(m["id"] == self.dmg_by for m in o.get("mechs", [])):
                self.glance_until = t + 0.9
            if t < self.glance_until:
                if any(m["id"] == self.dmg_by for m in o.get("mechs", [])):
                    self.glance_until = -9.0
                else:
                    a["aim"] = [self.dmg_bearing, 0.0]  # facing the attacker also takes away the rear bonus
        if "aim" in a:
            self.aim = a["aim"]
        gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), None)
        if gun and gun.get("reload", 0) == 0 and not a["fire"][0]:
            if gun.get("ammo", 99) == 0 or (primary is None and gun.get("ammo", 99) < gun["stats"]["mag"] * 0.6):
                a["reload"] = True
        return a

    def go_zone(self, a, nc, nr, margin, dn, hurry):
        p = self.me["pos"]
        k = max(0.0, nr - margin - 8.0) / max(dn, 1e-3)
        k = min(1.0, k)
        x, y = nc[0] + (p[0] - nc[0]) * k, nc[1] + (p[1] - nc[1]) * k
        self.nav(a, x, y, boost=hurry)

    def look(self, a, yaw=None):
        me = self.me
        if self.t - self.last_dmg_t < 2.5:
            a["aim"] = [self.dmg_bearing, 0.0]
            return
        if self.P["hear"] and self.t - self.snd_t < 1.5:
            a["aim"] = [self.snd_bearing, 0.0]
            return
        if yaw is None:
            mv = a.get("move", [0, 0])
            if math.hypot(mv[0], mv[1]) > 0.3:
                yaw = math.atan2(mv[1], mv[0]) + 1.0 * math.sin(self.t * 1.6)
            else:
                yaw = self.t * 1.3
        a["aim"] = [yaw, 0.0]

    def fight(self, a, primary, threats):
        me = self.me
        p = me["pos"]
        t = self.t
        q = primary["pos"]
        d = math.dist(p, q)
        bearing = math.atan2(q[1] - p[1], q[0] - p[0])
        ws = me["weapons"]
        gun = next((w for w in ws if w and w["kind"] == "gun"), None)
        laser = next((w for w in ws if w and w["kind"] == "laser"), None)
        ltng = next((w for w in ws if w and w["kind"] == "lightning"), None)
        gun_ready = bool(gun and gun.get("ammo", 0) > 0 and gun.get("reload", 0) == 0)
        laser_full = bool(laser and (laser.get("charge") or 0) >= 0.9 * laser["stats"].get("charge", 0.75))
        laser_soon = bool(laser and laser["cooldown"] <= 0.05)
        # is a laser pointed at me
        danger = False
        if self.P["laser_hide"]:
            # a charge shows for 0.7 s before the beam and cannot be kept past laser_hold: stay out of sight that long
            for m in threats:
                if t - m["seen"] < 0.15:
                    if m.get("charging"):
                        self.lz.setdefault(m["id"], t)
                    else:
                        self.lz.pop(m["id"], None)
            for i, t0 in list(self.lz.items()):
                m = self.known.get(i)
                if m is None or t - t0 > 6.0:
                    del self.lz[i]
                    continue
                b = math.atan2(p[1] - m["pos"][1], p[0] - m["pos"][0])
                if t - t0 <= self.P["laser_hold"] and abs(adiff(m["aim"][0], b)) < 0.6 and math.dist(m["pos"], p) < 290.0:
                    danger = True
        else:
            for m in threats:
                if m.get("charging") and t - m["seen"] < self.P["danger_mem"]:
                    b = math.atan2(p[1] - m["pos"][1], p[0] - m["pos"][0])
                    if abs(adiff(m["aim"][0], b)) < 0.25:
                        danger = True
        dodge = danger and bool(self.P["laser_hide"])
        if laser and not self.P["laser_disc"]:
            ready = laser_full or (gun_ready and not laser_soon)
        else:
            ready = gun_ready or laser_full
        if ltng and d < ltng["stats"]["range"]:
            ready = True
        crowd = sum(1 for m in threats if t - m["seen"] < 2.0 and math.dist(m["pos"], p) < self.P["crowd_r"])
        want_hide = self.retreat or danger or not ready or crowd >= self.P["crowd"]
        close = d < self.P["close"]
        if self.cover is None or t - self.cover_t > 0.3:
            c = self.find_cover(threats, primary, self.near, True)
            if c is None and (self.retreat or d > 45.0 or self.P["far_always"] or dodge):
                c = self.find_cover(threats, primary, self.far, False)
            self.cover = c
            self.cover_t = t
        c = None if NOCOVER else self.cover
        seen_now = t - primary["seen"] < 0.15 and self.can_hit(primary)
        if seen_now:
            self.blind_since = None
        elif self.blind_since is None:
            self.blind_since = t
        allow = not self.retreat or close or d < 70.0
        if self.retreat and self.P["quiet"]:
            allow = close or primary.get("hp", 1e9) < 0.25 * primary.get("max_hp", 1400)
        if c is not None and c["peek"] is not None:
            self.charge_ok = math.hypot(c["peek"][0] - p[0], c["peek"][1] - p[1]) <= self.P["peek_charge"]
        kite = None
        if self.P["ltng_gap"] and ltng is None:
            for m in threats:
                dm = math.dist(m["pos"], p)
                if "lightning" in (m.get("weapons") or []) and t - m["seen"] < 1.0 and dm < 65.0 and (kite is None or dm < kite[0]):
                    kite = (dm, m)
        if kite is not None:
            # lightning needs no aim inside 45 m: distance is the only defence
            dm, m = kite
            self.strafe(a, m, dm, math.atan2(m["pos"][1] - p[1], m["pos"][0] - p[0]), False)
            if dm < 55.0 and me["energy"] > 20:
                a["boost"] = True
        elif c is not None and not (close and seen_now and not self.retreat and c["peek"] is None):
            spot = c["hide"] if (want_hide or c["peek"] is None) else c["peek"]
            self.spot, self.spot_t = spot, t
            if c["straight"]:
                self.direct(a, spot[0], spot[1], stop=0.5)
                if dodge and me["energy"] > 10:
                    a["boost"] = True
            else:
                self.nav(a, spot[0], spot[1], spot[2], boost=self.retreat or dodge)
            # no peek spot and the enemy stays away: push after weak targets only
            weak = primary.get("hp", 1e9) < 0.35 * primary.get("max_hp", 1400)
            if c["peek"] is None and not want_hide and (weak or (self.blind_since and t - self.blind_since > 4.0 and not self.must_go)):
                if weak and me["hp"] > 0.5 * me["max_hp"]:
                    self.nav(a, q[0], q[1], q[2])
        else:
            self.strafe(a, primary, d, bearing, ltng is not None)
        if not dodge and kite is None:
            zc, zr = self.zone["center"], self.zone["radius"]
            near = None
            if self.P["drop_r"] > 0:
                for dr in self.drops_now:
                    qq = dr["pos"]
                    dd = math.hypot(qq[0] - p[0], qq[1] - p[1])
                    if dd < self.P["drop_r"] and math.hypot(qq[0] - zc[0], qq[1] - zc[1]) < zr - 8.0 \
                            and dr.get("lands_in", 0) < dd / 10.0 + 8.0 and (near is None or dd < near[0]):
                        near = (dd, qq, "drop")
            if near is None and self.P["grab"] > 0 and ws[1] is None and not self.retreat:
                for it in self.items.values():
                    if it["type"] != "pickup" or it.get("kind") in (None, "gun"):
                        continue
                    qq = it["pos"]
                    dd = math.hypot(qq[0] - p[0], qq[1] - p[1])
                    if dd < self.P["grab"] and math.hypot(qq[0] - zc[0], qq[1] - zc[1]) < zr - 8.0 and (near is None or dd < near[0]):
                        near = (dd, qq, "weapon")
            if near is None and self.P["cube_r"] > 0 and not self.retreat:
                for it in self.items.values():
                    if it["type"] != "cube":
                        continue
                    qq = it["pos"]
                    dd = math.hypot(qq[0] - p[0], qq[1] - p[1])
                    if dd < self.P["cube_r"] and math.hypot(qq[0] - zc[0], qq[1] - zc[1]) < zr - 8.0 and (near is None or dd < near[0]):
                        near = (dd, qq, "cube")
            if near is not None:
                # a second weapon about doubles our damage, a cube is +4% HP and damage, a drop is a full repair:
                # each is worth a walk under fire, and the torso keeps shooting on the way
                dd, qq, what = near
                if what == "drop" and dd < 6.0:
                    # stay inside the ring but keep moving
                    w = self.t * 1.6
                    self.direct(a, qq[0] + 3.5 * math.cos(w), qq[1] + 3.5 * math.sin(w), stop=0.3)
                else:
                    self.nav(a, qq[0], qq[1], qq[2], boost=True)
                    if dd < 6.0:
                        self.direct(a, qq[0], qq[1], stop=0.3)
        self.shoot(a, primary, allow=allow)
        # pre-charge the laser behind cover so the peek is one beam
        if dodge and laser:
            a["fire"][ws.index(laser)] = False  # a charge halves our speed, let it go and run
        elif laser and want_hide and not self.retreat and laser_soon and d < laser["stats"].get("range", 280):
            i = ws.index(laser)
            if (laser.get("charge") or 0) == 0 and (not self.P["laser_disc"] or self.charge_ok or seen_now):
                a["fire"][i] = True
        if not seen_now and "aim" in a and t - primary["seen"] > 1.0:
            a["aim"] = [bearing, a["aim"][1]]

    def strafe(self, a, primary, d, bearing, have_ltng):
        me = self.me
        p = me["pos"]
        if self.t > self.strafe_until:
            self.strafe_sign = self.rng.choice((-1, 1))
            self.strafe_until = self.t + self.rng.uniform(0.5, 1.3)
        ew = primary.get("weapons") or []
        want = self.P["range"]
        if self.P["laser_range"] > 0 and any(w and w["kind"] == "laser" for w in me["weapons"]):
            want = self.P["laser_range"]  # the beam hits at any range, their bullets do not
        if "lightning" in ew:
            want = 85.0
        if have_ltng and "lightning" not in ew:
            want = 30.0
        if self.retreat:
            want = 400.0
        rad = max(-1.0, min(1.0, (d - want) / 25.0))
        s = self.strafe_sign
        r1 = 1.0 if rad >= 0 else -1.0
        move = [0.0, 0.0]
        for ss, rr, L in ((s, rad, 5.0), (-s, rad, 5.0), (s, 0.0, 3.0), (-s, 0.0, 3.0), (0, r1, 3.0), (0, -r1, 3.0)):
            mx = -math.sin(bearing) * ss + math.cos(bearing) * rr
            my = math.cos(bearing) * ss + math.sin(bearing) * rr
            n = math.hypot(mx, my) or 1.0
            mx, my = mx / n, my / n
            if self.W.walkable_line(p[0], p[1], p[0] + mx * L, p[1] + my * L):
                move = [mx, my]
                if ss:
                    self.strafe_sign = ss
                break
        a["move"] = move
        if self.retreat and me["energy"] > 30:
            a["boost"] = True

    def loot(self, a, o, nc, nr, dn):
        me = self.me
        p = me["pos"]
        t = self.t
        z = self.zone
        ws = me["weapons"]
        held = {w["kind"] for w in ws if w}
        cur = PREF.get(ws[1]["kind"], 0) if ws[1] else -1
        limit_r = nr - 5.0 if (self.must_go or nr < 150) else None
        hpf = me["hp"] / max(1.0, me["max_hp"])
        best, bs = None, 1e9

        def inside(q):
            if math.hypot(q[0] - z["center"][0], q[1] - z["center"][1]) > z["radius"] - 10.0:
                return False
            return limit_r is None or math.hypot(q[0] - nc[0], q[1] - nc[1]) < limit_r

        for dr in o.get("drops", []):
            q = dr["pos"]
            d = math.hypot(q[0] - p[0], q[1] - p[1])
            if d < 260 and inside(q) and dr.get("lands_in", 0) < d / 8.0 + 20.0:
                s = d - 160.0
                if s < bs:
                    best, bs = ("drop", dr), s
        for it in self.items.values():
            q = it["pos"]
            d = math.hypot(q[0] - p[0], q[1] - p[1])
            if d > 170 or not inside(q):
                continue
            if it["type"] == "pickup":
                if it["kind"] in held or PREF.get(it["kind"], 0) <= cur:
                    continue
                s = d - 70.0
            elif it["type"] == "cube":
                s = d - 50.0
            else:
                s = d - 15.0
            if s < bs:
                best, bs = (it["type"], it), s
        if hpf > 0.7:
            for m in self.known.values():
                if m["kind"] != "npc":
                    continue
                d = math.dist(m["pos"], p)
                if d < 150 and inside(m["pos"]) and d + 15.0 < bs:
                    best, bs = ("npc", m), d + 15.0
        if best is None:
            self.hold(a, nc, nr, dn)
            return
        kind, it = best
        q = it["pos"]
        d = math.hypot(q[0] - p[0], q[1] - p[1])
        if kind == "drop":
            self.nav(a, q[0], q[1], q[2])
            self.look(a)
        elif kind in ("pickup", "cube"):
            self.nav(a, q[0], q[1], q[2])
            if d < 6.0:
                self.direct(a, q[0], q[1], stop=0.3)
            self.look(a)
        else:
            rec = dict(it)
            rec["kind"] = "crate" if kind == "crate" else "npc"
            rng = 85.0 if kind == "crate" else 75.0
            if d < rng and self.can_hit(rec):
                if kind == "npc":
                    self.strafe(a, rec, d, math.atan2(q[1] - p[1], q[0] - p[0]), False)
                self.shoot(a, rec)
            else:
                self.nav(a, q[0], q[1], q[2])
                self.look(a)

    def hold(self, a, nc, nr, dn):
        """Nothing to do: drift to a strong spot inside the next circle and watch."""
        p = self.me["pos"]
        best, bs = None, 1e9
        for f in self.forts:
            if f.get("jet", 0) > 20.0:
                continue
            if math.hypot(f["x"] - nc[0], f["y"] - nc[1]) > max(0.0, nr - 12.0):
                continue
            s = math.hypot(f["x"] - p[0], f["y"] - p[1]) + 300.0 * f.get("exposure", 0.5)
            if s < bs:
                best, bs = f, s
        if best is not None:
            x, y = best["x"], best["y"]
        else:
            k = min(1.0, (nr * 0.5) / max(dn, 1e-3))
            x, y = nc[0] + (p[0] - nc[0]) * k, nc[1] + (p[1] - nc[1]) * k
        if math.hypot(x - p[0], y - p[1]) > 4.0:
            self.nav(a, x, y)
        self.look(a)


def fallback(last):
    return {"move": last.get("move", [0, 0]), "aim": last.get("aim", [0, 0])}


def main():
    raw = sys.stdin.buffer
    hello = json.loads(raw.readline())
    try:
        bot = Bot(hello)
    except Exception:
        log(traceback.format_exc())
        bot = None
    out = sys.stdout
    out.write(json.dumps({"type": "ready", "name": "cornerstone"}) + "\n")
    out.flush()
    q = queue.Queue()

    def reader():
        fd = raw.fileno()
        buf = b""
        while True:
            chunk = os.read(fd, 1 << 18)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(b"\n")
            buf = parts.pop()
            if parts:
                q.put(parts)
        q.put(None)

    threading.Thread(target=reader, daemon=True).start()
    last = {}
    slow = 0
    while True:
        batch = q.get()
        if batch is None:
            break
        done = False
        while True:
            try:
                more = q.get_nowait()
            except queue.Empty:
                break
            if more is None:
                done = True
                break
            batch = batch + more
        line = None
        for ln in reversed(batch):
            if ln.strip():
                line = ln
                break
        if any(b'"type":"end"' in ln.replace(b" ", b"") for ln in batch[-2:]):
            break
        if line is None:
            if done:
                break
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if msg.get("type") == "end":
            break
        if msg.get("type") != "obs":
            continue
        t0 = time.perf_counter()
        try:
            act = bot.act(msg) if bot else {}
        except Exception:
            log(traceback.format_exc())
            act = fallback(last)
        act["tick"] = msg["tick"]
        out.write(json.dumps(act) + "\n")
        out.flush()
        last = act
        dt = time.perf_counter() - t0
        if dt > 0.03:
            slow += 1
            log(f"slow tick {msg['tick']} {dt * 1000:.0f} ms")
        if done:
            break


if __name__ == "__main__":
    main()
