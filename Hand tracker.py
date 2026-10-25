"""
Holoram - 3D Hand Control (v2)

Pinch to grab cubes with your webcam hand. Features:
  1. Grab offset      - cubes keep their offset from your hand (no teleporting)
  2. Pinch-point cursor - cursor blends from index tip to the pinch point
  3. One Euro filter  - low jitter when still, low lag when moving fast
  4. Floor grid, shadows and drop lines for depth perception
  5. Hover highlight + pinch progress ring
  6. Release inertia  - throw cubes; they drift, damp and bounce off the floor
  7. Two hands on one cube - stretch to scale, twist to rotate

Extras: threaded camera/tracking, pinch debounce, hand-lost grace period,
rotation-tolerant depth estimate with calibration.

Keys:  C = calibrate depth (hold hand at a comfortable distance)
       R = reset cubes      ESC = quit
"""

import math
import threading
import time

import cv2
import mediapipe as mp
import pygame

# ==================================================
# SETTINGS
# ==================================================
WIDTH, HEIGHT = 800, 600
CAM_WIDTH, CAM_HEIGHT = 640, 480
PREVIEW_W, PREVIEW_H = 240, 180

# Pinch detection (ratio of thumb-index distance to wrist->middle-knuckle)
GRAB_RATIO = 0.25
RELEASE_RATIO = 0.50
OPEN_RATIO = 0.90          # ratio at which the pinch ring starts to close
DEBOUNCE_FRAMES = 2        # consecutive tracking frames to change pinch state
HAND_LOST_GRACE = 0.3      # seconds a held cube survives tracking loss

# Depth
MIN_DEPTH = 3.0
MAX_DEPTH = 9.0
REF_DEPTH = 6.0            # depth at which the hand looks REF_HAND_SIZE px big
REF_HAND_SIZE = 110.0      # press C to calibrate this to your webcam

# Projection / world
FOCAL_LENGTH = 650.0
CUBE_SIZE = 0.65
FLOOR_Y = -1.2
MIN_SCALE, MAX_SCALE = 0.4, 2.5

# Physics
DAMPING = 0.92             # per 1/60 s
GRAVITY = 6.0              # try 6.0 if you want cubes to fall to the floor
RESTITUTION = 0.5
MAX_THROW_SPEED = 15.0

BG = (20, 23, 32)

# ==================================================
# MEDIAPIPE
# ==================================================
mp_hands = mp.solutions.hands


# ==================================================
# HELPERS
# ==================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def px_dist(a, b):
    """Distance between two normalized (x, y) points, in camera pixels."""
    return math.hypot((a[0] - b[0]) * CAM_WIDTH, (a[1] - b[1]) * CAM_HEIGHT)


def hand_size(p):
    """Rotation-tolerant hand size in camera pixels."""
    return max(px_dist(p[5], p[17]), px_dist(p[0], p[9]), px_dist(p[0], p[5]))


class OneEuro:
    """One Euro filter: heavy smoothing when slow, light smoothing when fast."""

    def __init__(self, min_cutoff=1.2, beta=0.03, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt):
        dt = max(dt, 1e-3)
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) / dt
        self.dx_prev += self._alpha(self.d_cutoff, dt) * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        self.x_prev += self._alpha(cutoff, dt) * (x - self.x_prev)
        return self.x_prev


# ==================================================
# 3D MATH / PROJECTION
# ==================================================
def project(p):
    x, y, z = p
    z = max(z, 0.1)
    return (int(WIDTH / 2 + x * FOCAL_LENGTH / z),
            int(HEIGHT / 2 - y * FOCAL_LENGTH / z))


def screen_to_world(sx, sy, depth):
    return ((sx - WIDTH / 2) * depth / FOCAL_LENGTH,
            (HEIGHT / 2 - sy) * depth / FOCAL_LENGTH,
            depth)


def rotate(v, roll, yaw):
    x, y, z = v
    cr, sr = math.cos(roll), math.sin(roll)
    x, y = x * cr - y * sr, x * sr + y * cr
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, z = x * cy + z * sy, -x * sy + z * cy
    return x, y, z


UNIT_VERTS = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
]

FACES = [
    (0, 1, 2, 3), (4, 5, 6, 7),
    (0, 1, 5, 4), (3, 2, 6, 7),
    (0, 3, 7, 4), (1, 2, 6, 5),
]

LIGHT = (0.4, 0.9, -0.5)
_n = math.sqrt(sum(c * c for c in LIGHT))
LIGHT = tuple(c / _n for c in LIGHT)


def new_cube(pos, color):
    return {
        "pos": list(pos), "vel": [0.0, 0.0, 0.0], "color": color,
        "scale": 1.0, "roll": 0.0, "yaw": 0.0,
        "mode": 0, "off": [0.0, 0.0, 0.0], "ref": {},
    }


def cube_half(c):
    return CUBE_SIZE / 2 * c["scale"]


def cube_world_vertices(c):
    h = cube_half(c)
    px, py, pz = c["pos"]
    out = []
    for v in UNIT_VERTS:
        x, y, z = rotate((v[0] * h, v[1] * h, v[2] * h), c["roll"], c["yaw"])
        out.append((px + x, py + y, pz + z))
    return out


# ==================================================
# THREADED CAMERA + HAND TRACKING
# ==================================================
class Tracker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.packet = None
        self.counter = 0
        self.running = True

    def opened(self):
        return self.cap.isOpened()

    def run(self):
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                stamp = time.perf_counter()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)

                found, used = [], set()
                if results.multi_hand_landmarks:
                    for lms, hd in zip(results.multi_hand_landmarks,
                                       results.multi_handedness):
                        label = hd.classification[0].label
                        if label in used:
                            label = "Left" if label == "Right" else "Right"
                        used.add(label)
                        found.append((label, [(p.x, p.y) for p in lms.landmark]))

                # Preview with skeleton so the user sees what the tracker sees
                preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
                for _, pts in found:
                    scaled = [(int(x * PREVIEW_W), int(y * PREVIEW_H))
                              for x, y in pts]
                    for a, b in mp_hands.HAND_CONNECTIONS:
                        cv2.line(preview, scaled[a], scaled[b], (255, 220, 80), 1)
                    for p in scaled:
                        cv2.circle(preview, p, 2, (80, 255, 160), -1)
                    # pinch points
                    cv2.circle(preview, scaled[4], 4, (0, 140, 255), -1)
                    cv2.circle(preview, scaled[8], 4, (0, 140, 255), -1)
                preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

                with self.lock:
                    self.counter += 1
                    self.packet = {
                        "id": self.counter, "t": stamp,
                        "hands": found, "preview": preview,
                    }
        finally:
            hands.close()
            self.cap.release()

    def latest(self):
        with self.lock:
            return self.packet

    def stop(self):
        self.running = False


# ==================================================
# HAND STATE
# ==================================================
class Hand:
    def __init__(self, name):
        self.name = name
        self.fx = OneEuro(1.0, 0.02)
        self.fy = OneEuro(1.0, 0.02)
        self.fz = OneEuro(0.6, 0.5)
        self.active = False
        self.visible = False
        self.lost = 0.0
        self.pinching = False
        self.grab_frames = 0
        self.release_frames = 0
        self.cube = None
        self.hover = None
        self.cx, self.cy = WIDTH / 2, HEIGHT / 2
        self.depth = REF_DEPTH
        self.ratio = 1.0
        self.closure = 0.0
        self.raw_size = REF_HAND_SIZE

    def reset_filters(self):
        self.fx.reset()
        self.fy.reset()
        self.fz.reset()

    def world(self):
        return screen_to_world(self.cx, self.cy, self.depth)


# ==================================================
# WORLD
# ==================================================
class World:
    def __init__(self):
        self.ref_size = REF_HAND_SIZE
        self.hands = {"Left": Hand("Left"), "Right": Hand("Right")}
        self.cubes = []
        self.reset_cubes()
        self.message = ""
        self.message_until = 0.0

    def reset_cubes(self):
        for h in self.hands.values():
            h.cube = None
            h.pinching = False
        self.cubes = [
            new_cube((-1.3, 0.7, 5.0), (245, 145, 55)),
            new_cube((0.0, 0.0, 6.0), (70, 200, 120)),
            new_cube((1.3, -0.7, 7.0), (85, 155, 255)),
        ]

    def holders(self, cube):
        return [h for h in self.hands.values() if h.cube is cube and h.pinching]

    def estimate_depth(self, size):
        # Apparent size is inversely proportional to distance.
        return clamp(REF_DEPTH * self.ref_size / max(size, 1.0),
                     MIN_DEPTH, MAX_DEPTH)

    def calibrate(self):
        sizes = [h.raw_size for h in self.hands.values() if h.visible]
        if sizes:
            self.ref_size = sum(sizes) / len(sizes)
            self.notify("Depth calibrated")
        else:
            self.notify("Show a hand to calibrate")

    def notify(self, text):
        self.message = text
        self.message_until = time.perf_counter() + 1.5

    # ---------------- picking / grabbing ----------------
    def find_cube(self, hand):
        best = None
        for c in self.cubes:
            holders = self.holders(c)
            if hand in holders or len(holders) >= 2:
                continue
            sx, sy = project(c["pos"])
            radius = 1.25 * FOCAL_LENGTH * cube_half(c) / max(c["pos"][2], 0.1) + 8
            if math.hypot(sx - hand.cx, sy - hand.cy) < radius:
                if best is None or c["pos"][2] < best["pos"][2]:
                    best = c
        return best

    def try_grab(self, hand):
        hand.pinching = True
        hand.cube = self.find_cube(hand)
        if hand.cube is not None:
            hand.cube["mode"] = 0  # force offset re-init

    def release(self, hand):
        hand.pinching = False
        hand.grab_frames = 0
        hand.release_frames = 0
        hand.cube = None  # cube keeps its smoothed velocity => inertia

    # ---------------- tracking update ----------------
    def process_hand(self, hand, pts, dt):
        if not hand.active:
            hand.reset_filters()
            hand.active = True
        hand.visible = True
        hand.lost = 0.0

        size = hand_size(pts)
        hand.raw_size = size
        ref = max(px_dist(pts[0], pts[9]), 1.0)
        hand.ratio = px_dist(pts[4], pts[8]) / ref
        hand.closure = clamp(
            (OPEN_RATIO - hand.ratio) / (OPEN_RATIO - GRAB_RATIO), 0.0, 1.0)

        # Cursor: index tip when open -> pinch midpoint when closed
        mid_x = (pts[4][0] + pts[8][0]) / 2
        mid_y = (pts[4][1] + pts[8][1]) / 2
        raw_x = lerp(pts[8][0], mid_x, hand.closure) * WIDTH
        raw_y = lerp(pts[8][1], mid_y, hand.closure) * HEIGHT
        hand.cx = hand.fx(clamp(raw_x, 0, WIDTH), dt)
        hand.cy = hand.fy(clamp(raw_y, 0, HEIGHT), dt)
        hand.depth = hand.fz(self.estimate_depth(size), dt)

        # Pinch state with debounce
        if not hand.pinching:
            hand.grab_frames = hand.grab_frames + 1 if hand.ratio < GRAB_RATIO else 0
            if hand.grab_frames >= DEBOUNCE_FRAMES:
                hand.grab_frames = 0
                self.try_grab(hand)
        else:
            hand.release_frames = (hand.release_frames + 1
                                   if hand.ratio > RELEASE_RATIO else 0)
            if hand.release_frames >= DEBOUNCE_FRAMES:
                self.release(hand)

        hand.hover = None if hand.pinching else self.find_cube(hand)

    def process_packet(self, packet, dt):
        seen = set()
        for label, pts in packet["hands"]:
            if label in self.hands:
                seen.add(label)
                self.process_hand(self.hands[label], pts, dt)

        for name, hand in self.hands.items():
            if name in seen:
                continue
            hand.visible = False
            hand.hover = None
            if hand.active:
                hand.lost += dt
                if hand.lost > HAND_LOST_GRACE:
                    self.release(hand)
                    hand.active = False

        self.update_held(dt)

    def update_held(self, dt):
        for c in self.cubes:
            hs = sorted(self.holders(c), key=lambda h: h.name)
            n = len(hs)
            if n == 0:
                c["mode"] = 0
                continue

            old = c["pos"][:]

            if n == 1:
                hw = hs[0].world()
                if c["mode"] != 1:
                    c["mode"] = 1
                    c["off"] = [c["pos"][i] - hw[i] for i in range(3)]
                c["pos"] = [hw[i] + c["off"][i] for i in range(3)]
            else:
                a, b = hs
                wa, wb = a.world(), b.world()
                mid = [(wa[i] + wb[i]) / 2 for i in range(3)]
                dist = max(math.hypot(a.cx - b.cx, a.cy - b.cy), 30.0)
                ang = math.atan2(-(b.cy - a.cy), b.cx - a.cx)
                dz = a.depth - b.depth
                if c["mode"] != 2:
                    c["mode"] = 2
                    c["off"] = [c["pos"][i] - mid[i] for i in range(3)]
                    c["ref"] = {
                        "dist": dist, "last_ang": ang, "dz": dz,
                        "scale": c["scale"], "yaw": c["yaw"],
                    }
                r = c["ref"]
                c["pos"] = [mid[i] + c["off"][i] for i in range(3)]
                c["scale"] = clamp(r["scale"] * dist / r["dist"],
                                   MIN_SCALE, MAX_SCALE)
                c["roll"] += wrap_angle(ang - r["last_ang"])
                r["last_ang"] = ang
                c["yaw"] = r["yaw"] + (dz - r["dz"]) * 0.5

            c["pos"][2] = clamp(c["pos"][2], MIN_DEPTH, MAX_DEPTH)
            c["pos"][1] = max(c["pos"][1], FLOOR_Y + cube_half(c))

            inst = [(c["pos"][i] - old[i]) / max(dt, 1e-3) for i in range(3)]
            c["vel"] = [0.5 * c["vel"][i] + 0.5 * inst[i] for i in range(3)]

    # ---------------- physics ----------------
    def physics(self, dt):
        for c in self.cubes:
            if self.holders(c):
                continue

            vel, pos = c["vel"], c["pos"]
            speed = math.sqrt(sum(v * v for v in vel))
            if speed > MAX_THROW_SPEED:
                k = MAX_THROW_SPEED / speed
                vel[:] = [v * k for v in vel]
            elif speed < 0.05:
                vel[:] = [0.0, 0.0, 0.0]

            vel[1] -= GRAVITY * dt
            for i in range(3):
                pos[i] += vel[i] * dt
            damp = DAMPING ** (dt * 60)
            vel[:] = [v * damp for v in vel]

            half = cube_half(c)

            # Floor bounce
            if pos[1] - half < FLOOR_Y:
                pos[1] = FLOOR_Y + half
                if vel[1] < 0:
                    vel[1] = -vel[1] * RESTITUTION
                    if abs(vel[1]) < 0.3:
                        vel[1] = 0.0
                vel[0] *= 0.9
                vel[2] *= 0.9

            # Depth bounds
            if pos[2] < MIN_DEPTH:
                pos[2], vel[2] = MIN_DEPTH, abs(vel[2]) * 0.5
            elif pos[2] > MAX_DEPTH:
                pos[2], vel[2] = MAX_DEPTH, -abs(vel[2]) * 0.5

            # Keep inside the visible frustum
            lim_x = (WIDTH / 2 - 20) * pos[2] / FOCAL_LENGTH
            lim_y = (HEIGHT / 2 - 20) * pos[2] / FOCAL_LENGTH
            if pos[0] < -lim_x:
                pos[0], vel[0] = -lim_x, abs(vel[0]) * 0.5
            elif pos[0] > lim_x:
                pos[0], vel[0] = lim_x, -abs(vel[0]) * 0.5
            if pos[1] > lim_y:
                pos[1], vel[1] = lim_y, -abs(vel[1]) * 0.5


# ==================================================
# RENDERING
# ==================================================
def draw_floor(screen):
    horizon = HEIGHT // 2
    pygame.draw.line(screen, (45, 52, 68), (0, horizon), (WIDTH, horizon), 1)

    z_near, z_far = 2.0, 10.0
    # Lines running into the distance
    for xi in range(-10, 11):
        a = project((xi, FLOOR_Y, z_near))
        b = project((xi, FLOOR_Y, z_far))
        pygame.draw.line(screen, (38, 46, 62), a, b, 1)
    # Lines across
    for zi in range(int(z_near), int(z_far) + 1):
        a = project((-10, FLOOR_Y, zi))
        b = project((10, FLOOR_Y, zi))
        fade = (zi - z_near) / (z_far - z_near)
        pygame.draw.line(screen, lerp_color((60, 70, 92), BG, fade * 0.8), a, b, 1)


def draw_shadows(screen, overlay, cubes):
    overlay.fill((0, 0, 0, 0))
    for c in cubes:
        half = cube_half(c)
        x, y, z = c["pos"]
        height = max((y - half) - FLOOR_Y, 0.0)
        alpha = int(clamp(150 - height * 45, 25, 150))
        s = half * 1.15
        quad = [project((x - s, FLOOR_Y, z - s)), project((x + s, FLOOR_Y, z - s)),
                project((x + s, FLOOR_Y, z + s)), project((x - s, FLOOR_Y, z + s))]
        pygame.draw.polygon(overlay, (0, 0, 0, alpha), quad)
    screen.blit(overlay, (0, 0))


def draw_drop_lines(screen, cubes, held_ids, hover_ids):
    for c in cubes:
        half = cube_half(c)
        x, y, z = c["pos"]
        top = project((x, y - half, z))
        bottom = project((x, FLOOR_Y, z))
        if id(c) in held_ids:
            color, w = (255, 255, 255), 2
        elif id(c) in hover_ids:
            color, w = (120, 230, 255), 1
        else:
            color, w = (60, 70, 92), 1
        pygame.draw.line(screen, color, top, bottom, w)


def draw_cube(screen, c, state):
    verts = cube_world_vertices(c)
    proj = [project(v) for v in verts]
    center = c["pos"]

    if state == "two":
        edge_color, edge_w = (255, 200, 60), 3
    elif state == "held":
        edge_color, edge_w = (255, 255, 255), 3
    elif state == "hover":
        edge_color, edge_w = (120, 230, 255), 3
    else:
        edge_color, edge_w = (30, 32, 40), 1

    fog = clamp((center[2] - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH), 0, 1) * 0.55

    faces = []
    for face in FACES:
        fc = [sum(verts[i][k] for i in face) / 4 for k in range(3)]
        n = [fc[k] - center[k] for k in range(3)]
        length = math.sqrt(sum(v * v for v in n)) or 1.0
        n = [v / length for v in n]
        # Back-face culling (camera at origin)
        if sum(n[k] * fc[k] for k in range(3)) >= 0:
            continue
        light = max(0.0, sum(n[k] * LIGHT[k] for k in range(3)))
        shade = 0.5 + 0.5 * light
        color = tuple(int(clamp(ch * shade, 0, 255)) for ch in c["color"])
        color = lerp_color(color, BG, fog)
        faces.append((fc[2], face, color))

    faces.sort(key=lambda f: f[0], reverse=True)
    for _, face, color in faces:
        pts = [proj[i] for i in face]
        pygame.draw.polygon(screen, color, pts)
        pygame.draw.polygon(screen, edge_color, pts, edge_w)


def draw_cursor(screen, hand):
    if not hand.visible:
        return
    if hand.cube is not None:
        color = (255, 190, 60)
    elif hand.pinching:
        color = (255, 255, 255)
    else:
        color = (120, 230, 255)

    pos = (int(hand.cx), int(hand.cy))
    radius = int(10 + (1.0 - hand.closure) * 24)
    pygame.draw.circle(screen, color, pos, radius, 2)
    pygame.draw.circle(screen, color, pos, 4)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Holoram - 3D Hand Control")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    small = pygame.font.Font(None, 20)
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    tracker = Tracker()
    if not tracker.opened():
        pygame.quit()
        raise RuntimeError("Could not open webcam.")
    tracker.start()

    world = World()
    last_id = -1
    last_t = None
    preview_surface = None

    running = True
    try:
        while running:
            dt = min(clock.tick(60) / 1000.0, 0.05)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c:
                        world.calibrate()
                    elif event.key == pygame.K_r:
                        world.reset_cubes()

            # New tracking data?
            packet = tracker.latest()
            if packet is not None and packet["id"] != last_id:
                track_dt = (1 / 30 if last_t is None
                            else clamp(packet["t"] - last_t, 0.005, 0.2))
                last_id, last_t = packet["id"], packet["t"]
                world.process_packet(packet, track_dt)
                preview_surface = pygame.surfarray.make_surface(
                    packet["preview"].swapaxes(0, 1))

            world.physics(dt)

            # ---------------- render ----------------
            screen.fill(BG)
            draw_floor(screen)
            draw_shadows(screen, overlay, world.cubes)

            held_ids = {id(h.cube) for h in world.hands.values()
                        if h.cube is not None and h.pinching}
            hover_ids = {id(h.hover) for h in world.hands.values()
                         if h.hover is not None}

            draw_drop_lines(screen, world.cubes, held_ids, hover_ids)

            for c in sorted(world.cubes, key=lambda k: k["pos"][2], reverse=True):
                holders = world.holders(c)
                if len(holders) >= 2:
                    state = "two"
                elif holders:
                    state = "held"
                elif id(c) in hover_ids:
                    state = "hover"
                else:
                    state = "normal"
                draw_cube(screen, c, state)

            for hand in world.hands.values():
                draw_cursor(screen, hand)

            # Status text
            y = 16
            any_visible = False
            for name in ("Left", "Right"):
                h = world.hands[name]
                if not h.visible:
                    continue
                any_visible = True
                if h.cube is not None:
                    status = f"{name}: HOLDING | depth {h.depth:.1f}"
                elif h.pinching:
                    status = f"{name}: PINCHING - move over a cube"
                elif h.hover is not None:
                    status = f"{name}: pinch to grab"
                else:
                    status = f"{name}: hand detected"
                screen.blit(font.render(status, True, (235, 235, 245)), (20, y))
                y += 22
            if not any_visible:
                screen.blit(font.render("NO HAND", True, (235, 235, 245)), (20, y))

            if time.perf_counter() < world.message_until:
                screen.blit(font.render(world.message, True, (255, 200, 60)),
                            (20, y + 22))

            hint = ("Pinch=grab | 2 hands on one cube: stretch=scale, twist=rotate | "
                    "C calibrate | R reset | ESC quit")
            screen.blit(small.render(hint, True, (140, 148, 168)), (20, HEIGHT - 26))

            # Webcam preview
            if preview_surface is not None:
                screen.blit(preview_surface, (WIDTH - PREVIEW_W - 10, 40))
                pygame.draw.rect(screen, (230, 230, 240),
                                 (WIDTH - PREVIEW_W - 10, 40, PREVIEW_W, PREVIEW_H), 2)

            pygame.display.flip()
    finally:
        tracker.stop()
        tracker.join(timeout=1.0)
        pygame.quit()


if __name__ == "__main__":
    main()
