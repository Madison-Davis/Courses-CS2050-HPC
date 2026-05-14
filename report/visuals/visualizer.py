#!/usr/bin/env python3
"""
Orbital Body Visualizer
========================
Reads serial_positions.csv and serial_timing.csv and renders each body
as a glowing sphere moving through 3D black space.

Controls:
  Left-drag     — orbit camera
  Right-drag    — pan
  Scroll wheel  — zoom
  ESC / Q       — quit
"""

import sys, math, os, time, collections
import numpy as np
import pandas as pd
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# ─── CONFIG ──────────────────────────────────────────────────────────────────
POSITIONS_FILE = "serial_positions.csv"
TIMING_FILE    = "serial_timing.csv"
WINDOW_W, WINDOW_H = 1280, 800
FPS            = 60
PLAYBACK_FPS   = 30         # simulation steps per second
TRAIL_LENGTH   = 60         # positions to keep in trail
SPHERE_SLICES  = 20

PALETTE = [
    (0.0,  1.0,  0.67),   # cyan-green
    (0.0,  0.8,  1.0),    # sky blue
    (1.0,  0.4,  0.0),    # orange
    (1.0,  0.0,  0.67),   # magenta
    (1.0,  1.0,  0.0),    # yellow
    (0.67, 0.0,  1.0),    # violet
    (1.0,  0.27, 0.27),   # red
    (0.27, 1.0,  1.0),    # aqua
]

# ─── DATA LOADING ────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(POSITIONS_FILE):
        # Demo data so the script is runnable standalone
        print(f"[warn] {POSITIONS_FILE} not found — using built-in demo data")
        from io import StringIO
        pos_txt = """time,body,x,y,z
0,0,5.41775e+06,1.22028e+07,747635
0,1,-1.10658e+07,3.91249e+06,-941038
0,2,-5.3504e+06,9.20473e+06,-761759"""
        tim_txt = "step,time_sec\n0,0.114772"
        pos_df = pd.read_csv(StringIO(pos_txt))
        tim_df = pd.read_csv(StringIO(tim_txt))
    else:
        pos_df = pd.read_csv(POSITIONS_FILE)
        tim_df = pd.read_csv(TIMING_FILE) if os.path.exists(TIMING_FILE) else pd.DataFrame(columns=["step","time_sec"])

    # Group positions by time
    steps = {}
    for t, grp in pos_df.groupby("time"):
        steps[float(t)] = grp[["body","x","y","z"]].to_dict("records")
    sorted_times = sorted(steps.keys())

    # Timing lookup: step index → time_sec
    # Drop any rows where 'step' is not a plain integer (e.g. summary rows like "total")
    if len(tim_df):
        tim_df = tim_df[pd.to_numeric(tim_df["step"], errors="coerce").notna()].copy()
        tim_df["step"] = tim_df["step"].astype(float).astype(int)
        timing = dict(zip(tim_df["step"], pd.to_numeric(tim_df["time_sec"], errors="coerce")))
    else:
        timing = {}

    # Normalise coords to ~[-10, 10] scene units
    all_xyz = pos_df[["x","y","z"]].values
    max_val = np.abs(all_xyz).max()
    scale = 10.0 / max_val if max_val else 1.0

    all_body_ids = sorted(pos_df["body"].unique())
    return steps, sorted_times, timing, scale, all_body_ids

# ─── OPENGL HELPERS ──────────────────────────────────────────────────────────
def draw_sphere(r, slices):
    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)
    gluSphere(quad, r, slices, slices)
    gluDeleteQuadric(quad)

def draw_circle_xz(radius, segs=64):
    glBegin(GL_LINE_LOOP)
    for i in range(segs):
        a = 2*math.pi*i/segs
        glVertex3f(radius*math.cos(a), 0, radius*math.sin(a))
    glEnd()

def draw_starfield(stars):
    glPointSize(1.5)
    glColor4f(1,1,1,0.6)
    glBegin(GL_POINTS)
    for s in stars:
        glVertex3f(*s)
    glEnd()

def draw_grid(size=20, step=2):
    glLineWidth(1)
    glColor4f(0.0, 0.2, 0.12, 0.5)
    glBegin(GL_LINES)
    for i in range(-size, size+1, step):
        glVertex3f(i, -8, -size)
        glVertex3f(i, -8,  size)
        glVertex3f(-size, -8, i)
        glVertex3f( size, -8, i)
    glEnd()

def draw_axes():
    glLineWidth(1.5)
    axes = [(1,0,0,0.4,0,0), (0,1,0,0,0.4,0), (0,0,1,0,0,0.4)]
    for r,g,b,dx,dy,dz in axes:
        glColor4f(r,g,b,0.3)
        glBegin(GL_LINES)
        glVertex3f(0,0,0); glVertex3f(dx*30,dy*30,dz*30)
        glEnd()

def draw_trail(trail, color, alpha_max=0.55):
    n = len(trail)
    if n < 2:
        return
    glLineWidth(1.5)
    glBegin(GL_LINE_STRIP)
    for i, pos in enumerate(trail):
        a = alpha_max * (i / (n-1)) ** 1.5
        glColor4f(*color, a)
        glVertex3f(*pos)
    glEnd()

# ─── HUD TEXT ────────────────────────────────────────────────────────────────
def init_font():
    pygame.font.init()
    try:
        return pygame.font.SysFont("monospace", 16)
    except:
        return pygame.font.Font(None, 18)

def render_hud(surface, font, sim_time, n_bodies, step_idx, total_steps,
               timestep, loop_count, frame_num):
    """Blit HUD text onto a pygame surface (drawn in 2D after 3D pass)."""
    FG  = (0, 255, 170)
    DIM = (0, 130, 80)
    H   = surface.get_height()
    W   = surface.get_width()

    def txt(s, color=FG):
        return font.render(s, True, color)

    # Bottom bar background
    bar_h = 54
    bar_surf = pygame.Surface((W, bar_h), pygame.SRCALPHA)
    bar_surf.fill((0, 0, 0, 200))
    surface.blit(bar_surf, (0, H - bar_h))

    # Separator line
    pygame.draw.line(surface, (0, 80, 50), (0, H-bar_h), (W, H-bar_h), 1)

    # Progress bar
    prog = (step_idx / max(total_steps-1, 1))
    pygame.draw.rect(surface, (0, 40, 25), (0, H-bar_h-2, W, 2))
    pygame.draw.rect(surface, (0, 255, 170), (0, H-bar_h-2, int(W*prog), 2))

    # Bottom labels
    labels = [
        ("SIMULATION TIME", f"{sim_time:.6f} s"),
        ("BODIES ALIVE",    f"{n_bodies}"),
        ("TIMESTEP",        f"{timestep:.6f} s" if timestep is not None else "—"),
    ]
    col_w = W // len(labels)
    for i, (lbl, val) in enumerate(labels):
        lx = i * col_w + col_w // 2
        ls = font.render(lbl, True, (0, 100, 60))
        vs = pygame.font.SysFont("monospace", 20, bold=True).render(val, True, FG)
        surface.blit(ls, ls.get_rect(centerx=lx, y=H-bar_h+6))
        surface.blit(vs, vs.get_rect(centerx=lx, y=H-bar_h+22))

    # Top-left
    surface.blit(txt(f"FRAME  {frame_num:04d}", DIM), (16, 16))
    surface.blit(txt(f"STEP   {step_idx}", DIM),      (16, 34))

    # Top-right
    live = txt("● LIVE PLAYBACK", FG)
    lp   = txt(f"LOOP   {loop_count}", DIM)
    surface.blit(live, (W - live.get_width() - 16, 16))
    surface.blit(lp,   (W - lp.get_width()   - 16, 34))

    # Controls (bottom-right above bar)
    hints = ["LEFT DRAG · ROTATE", "RIGHT DRAG · PAN", "SCROLL · ZOOM", "ESC · QUIT"]
    for i, h in enumerate(reversed(hints)):
        hs = font.render(h, True, (0, 70, 45))
        surface.blit(hs, (W - hs.get_width() - 14, H - bar_h - 18 - i*16))

# ─── CAMERA ──────────────────────────────────────────────────────────────────
class Camera:
    def __init__(self):
        self.theta   = 0.4
        self.phi     = 0.55      # angled — above the ring, looking down
        self.radius  = 28.0
        self.pan_x   = 0.0
        self.pan_y   = 0.0
        # smooth targets
        self.t_theta  = 0.4
        self.t_phi    = 0.55
        self.t_radius = self.radius
        self.t_pan_x  = self.pan_x
        self.t_pan_y  = self.pan_y

    def smooth(self, k=0.12):
        self.theta  += (self.t_theta  - self.theta)  * k
        self.phi    += (self.t_phi    - self.phi)    * k
        self.radius += (self.t_radius - self.radius) * k
        self.pan_x  += (self.t_pan_x  - self.pan_x)  * k
        self.pan_y  += (self.t_pan_y  - self.pan_y)  * k
        self.phi = max(0.05, min(math.pi - 0.05, self.phi))

    def apply(self):
        eye_x = self.pan_x + self.radius * math.sin(self.phi) * math.sin(self.theta)
        eye_y = self.pan_y + self.radius * math.cos(self.phi)
        eye_z =              self.radius * math.sin(self.phi) * math.cos(self.theta)
        gluLookAt(eye_x, eye_y, eye_z,
                  self.pan_x, self.pan_y, 0,
                  0, 1, 0)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    steps, sorted_times, timing, scale, all_body_ids = load_data()

    pygame.init()
    pygame.display.set_caption("Orbital Body Visualizer")
    flags = DOUBLEBUF | OPENGL | RESIZABLE
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), flags)
    clock  = pygame.time.Clock()
    font   = init_font()

    # OpenGL initial state
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_POINT_SMOOTH)

    # Lighting
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [0, 15, 10, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1, 1, 1, 1])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1, 1, 1, 1])
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.15, 0.15, 0.2, 1])

    # Random starfield
    rng   = np.random.default_rng(42)
    stars = (rng.random((3000, 3)) - 0.5) * 400

    # Per-body state
    trails   = {bid: collections.deque(maxlen=TRAIL_LENGTH) for bid in all_body_ids}
    cam      = Camera()
    hud_surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)

    # Playback state
    step_idx   = 0
    loop_count = 0
    frame_num  = 0
    last_step_t = time.time()
    step_interval = 1.0 / PLAYBACK_FPS

    # Mouse state
    mouse_down  = False
    right_down  = False
    last_mx     = 0
    last_my     = 0

    def resize(w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, w/h, 0.1, 2000)
        glMatrixMode(GL_MODELVIEW)

    resize(WINDOW_W, WINDOW_H)

    running = True
    while running:
        W, H = screen.get_size()

        # ── Events ──────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key in (K_ESCAPE, K_q):
                    running = False
            elif event.type == VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, flags)
                hud_surf = pygame.Surface(event.size, pygame.SRCALPHA)
                resize(event.w, event.h)

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down = True; right_down = False
                elif event.button == 3:
                    mouse_down = True; right_down = True
                elif event.button == 4:   # scroll up → zoom in
                    cam.t_radius = max(2, cam.t_radius - 1.5)
                elif event.button == 5:   # scroll down → zoom out
                    cam.t_radius = min(200, cam.t_radius + 1.5)
                last_mx, last_my = event.pos

            elif event.type == MOUSEBUTTONUP:
                if event.button in (1, 3):
                    mouse_down = False

            elif event.type == MOUSEMOTION and mouse_down:
                dx = event.pos[0] - last_mx
                dy = event.pos[1] - last_my
                last_mx, last_my = event.pos
                if right_down:
                    cam.t_pan_x -= dx * 0.025
                    cam.t_pan_y += dy * 0.025
                else:
                    cam.t_theta -= dx * 0.008
                    cam.t_phi   += dy * 0.008

        # ── Advance simulation step ──────────────────────────────────────────
        now = time.time()
        if now - last_step_t >= step_interval:
            last_step_t = now
            step_idx += 1
            frame_num += 1
            if step_idx >= len(sorted_times):
                step_idx   = 0
                loop_count += 1
                for bid in all_body_ids:
                    trails[bid].clear()

        t          = sorted_times[step_idx]
        bodies_now = steps[t]
        active_ids = {b["body"] for b in bodies_now}

        for b in bodies_now:
            bid = b["body"]
            pos = (b["x"]*scale, b["y"]*scale, b["z"]*scale)
            trails[bid].append(pos)

        timestep = timing.get(step_idx, None)

        # ── 3-D render pass ─────────────────────────────────────────────────
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        cam.smooth()
        cam.apply()

        # Starfield (no lighting)
        glDisable(GL_LIGHTING)
        draw_starfield(stars)
        draw_grid()
        draw_axes()

        pulse = 0.7 + 0.3 * math.sin(now * 3.0)

        # ── Central planet ───────────────────────────────────────────────────
        glPushMatrix()
        # Outer atmosphere glow rings
        glDisable(GL_LIGHTING)
        glDepthMask(False)
        glColor4f(0.6, 0.6, 0.6, 0.07 * pulse)
        draw_circle_xz(1.0)
        glColor4f(0.6, 0.6, 0.6, 0.05 * pulse)
        draw_circle_xz(1.4)
        glColor4f(0.6, 0.6, 0.6, 0.03 * pulse)
        draw_circle_xz(1.8)
        glDepthMask(True)
        # Planet sphere
        glEnable(GL_LIGHTING)
        glMaterialfv(GL_FRONT, GL_EMISSION,  [0.2*pulse, 0.2*pulse, 0.2*pulse, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE,   [0.55, 0.55, 0.55, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR,  [0.8, 0.8, 0.8, 1.0])
        glMaterialf (GL_FRONT, GL_SHININESS, 90)
        draw_sphere(0.55, SPHERE_SLICES)
        glPopMatrix()

        for i, bid in enumerate(all_body_ids):
            color = PALETTE[i % len(PALETTE)]
            tr    = list(trails[bid])

            # Trail
            if len(tr) >= 2:
                draw_trail(tr, color)

            # Sphere (only if alive this step)
            if bid in active_ids and tr:
                px, py, pz = tr[-1]
                glPushMatrix()
                glTranslatef(px, py, pz)

                # Outer glow halo (blended additive)
                glDisable(GL_LIGHTING)
                glDepthMask(False)
                glColor4f(*color, 0.08 * pulse)
                draw_circle_xz(0.35)
                glColor4f(*color, 0.06 * pulse)
                draw_circle_xz(0.55)
                glDepthMask(True)

                # Lit sphere — small grey dot
                glEnable(GL_LIGHTING)
                em = [0.35*pulse, 0.35*pulse, 0.35*pulse, 1.0]
                dif = [0.55, 0.55, 0.55, 1.0]
                glMaterialfv(GL_FRONT, GL_EMISSION,  em)
                glMaterialfv(GL_FRONT, GL_DIFFUSE,   dif)
                glMaterialfv(GL_FRONT, GL_SPECULAR,  [0.8,0.8,0.8,1])
                glMaterialf (GL_FRONT, GL_SHININESS, 60)
                draw_sphere(0.07, SPHERE_SLICES)

                glPopMatrix()

        # ── 2-D HUD overlay ──────────────────────────────────────────────────
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

        hud_surf = pygame.Surface((W, H), pygame.SRCALPHA)
        hud_surf.fill((0, 0, 0, 0))
        render_hud(hud_surf, font,
                   sim_time   = t,
                   n_bodies   = len(active_ids),
                   step_idx   = step_idx,
                   total_steps= len(sorted_times),
                   timestep   = timestep,
                   loop_count = loop_count,
                   frame_num  = frame_num)

        # Blit HUD via a temporary pygame surface → pixel read → texture
        hud_data = pygame.image.tostring(hud_surf, "RGBA", True)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix(); glLoadIdentity()
        glOrtho(0, W, 0, H, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix(); glLoadIdentity()

        glRasterPos2i(0, 0)
        glDrawPixels(W, H, GL_RGBA, GL_UNSIGNED_BYTE, hud_data)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_DEPTH_TEST)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()