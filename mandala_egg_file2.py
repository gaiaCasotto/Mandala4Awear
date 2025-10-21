import taichi as ti
import math
import time
ti.init(arch=ti.gpu)  # or ti.cpu

# --- Config ---
W, H = 800, 800

# --- Buffers ---
img = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))

vec2 = ti.types.vector(2, ti.f32)
vec3 = ti.types.vector(3, ti.f32)


# Uniforms / controls
time_mul     = ti.field(dtype=ti.f32, shape=())
freq_mul     = ti.field(dtype=ti.f32, shape=())
contrast     = ti.field(dtype=ti.f32, shape=())
heart_weight = ti.field(dtype=ti.f32, shape=())
octaves      = ti.field(dtype=ti.i32, shape=())
phase_jitter = ti.field(dtype=ti.f32, shape=())
speed        = ti.field(dtype=ti.f32, shape=())

palette_bias = ti.Vector.field(3, ti.f32, shape=())


time_mul[None]     = 0.6
freq_mul[None]     = 1.0
contrast[None]     = 0.9
heart_weight[None] = 0.12
phase_jitter[None] = 0.0
octaves[None]      = 3
speed[None]        = 1.0

palette_bias[None] = ti.Vector([0.0, 0.0, 0.0])  # no tint


# --- Helpers ---
@ti.func
def fract1(x: ti.f32) -> ti.f32:
    return x - ti.floor(x)

@ti.func
def fract2(v: vec2) -> vec2:
    return vec2(fract1(v.x), fract1(v.y))

@ti.func
def length2(v: vec2) -> ti.f32:
    return ti.sqrt(v.dot(v))

@ti.func
def palette(t: ti.f32) -> vec3:
    a = vec3(0.5, 0.5, 0.5)
    b = vec3(0.5, 0.5, 0.5)
    c = vec3(1.0, 1.0, 1.0)
    d = vec3(0.263, 0.416, 0.557)
    # a + b * cos(2π * (c * t + d))
    two_pi = 6.28318
    base = a + b * ti.cos(two_pi * (c * t + d))
    palette = base + palette_bias[None]
    return palette

@ti.func
def sdHeart(p_in: vec2) -> ti.f32:
    # p.x = abs(p.x)
    p = vec2(ti.abs(p_in.x), p_in.y)
    #sqrt = 0 this loses precision: i32 <- f32
    sqrt: ti.f32 = 0.0

    if p.y + p.x > 1.0:
        q = p - vec2(0.25, 0.75)
        sqrt = ti.sqrt(q.dot(q)) - ti.sqrt(2.0) * 0.25
    else:
        q1 = p - vec2(0.0, 1.0)
        dot1 = q1.dot(q1)

        m = ti.max(p.x + p.y, 0.0) * 0.5
        q2 = p - vec2(m, m)
        dot2 = q2.dot(q2)

        dmin = ti.min(dot1, dot2)
        sqrt = ti.sqrt(dmin) * ti.math.sign(p.x - p.y)
    
    return sqrt


#----------------- REnder -----------------
@ti.kernel
def render(iTime: ti.f32):
    tm = time_mul[None]
    fm = freq_mul[None]
    cw = contrast[None]
    hw = heart_weight[None]
    octs = octaves[None]
    jitter = phase_jitter[None]


    for x, y in img:
        # fragCoord in [0, W/H)
        frag = vec2(ti.cast(x, ti.f32), ti.cast(y, ti.f32))
        iRes = vec2(float(W), float(H))
        # uv like Shadertoy: (-aspect/2..aspect/2, -0.5..0.5) with division by height
        uv = (frag * 2.0 - iRes) / iRes.y
        uv0 = uv
        color = vec3(0.0, 0.0, 0.0)

        t = iTime * tm + ti.sin(iTime * 3.7) * jitter * 0.03
        
        for i in range(4):
            if i >= octs:
                break #hard cap
            fi = ti.cast(i, ti.f32)

            uvt = fract2(uv * (1.5 * fm)) - vec2(0.5, 0.5)

            d = length2(uvt) * ti.exp(-length2(uv0))
            d += sdHeart(uvt) * (0.1 * hw)

            col = palette(length2(uv0) + fi * 0.4 + t * 0.4)

            d = ti.sin(d * 8.0 * fm + t) / 8.0
            d = ti.abs(d)

            # avoid div-by-zero blowups; keep the shader feel
            eps = 1e-4
            d = ti.pow(0.01 / ti.max(d, eps), 1.2)

            color += col * d
            uv = uvt

        #(higher cw = punchier/“harsher”)
        color = color / (1.0 + cw * color)  # simple Reinhard
        img[x, y] = ti.min(color, vec3(1.0, 1.0, 1.0))


#------------------ SMOOTHING between states -----------------------
#smoothing with EMA to avoid rapid flips of state -> rapid flips of parameters
def state_to_level(state: str) -> float:
    return STATE_LEVEL.get(norm_state(state), 0.0)


class SoftStateSmoother:
    """
    Blends incoming discrete states into a smoothed scalar level with hysteretic labeling.
    - EMA toward target level (asymmetric tau).
    - Hysteresis thresholds to produce a stable label from the smoothed level.
    """
    def __init__(self, tau_rise=0.25, tau_fall=0.80, margin=0.1):
        self.level = 0.0            # smoothed scalar in [0..1]
        self.label = "calm"         # stable label derived from self.level
        self.tau_rise = tau_rise
        self.tau_fall = tau_fall
        self.margin = margin        # hysteresis band around thresholds

        # base thresholds between classes (without hysteresis)
        self.t_calm_mod  = 0.16
        self.t_mod_high  = 0.50
        self.t_high_ext  = 0.83

    def _exp_smooth(self, current, target, dt, tau):
        if tau <= 0.0:
            return target
        k = 1.0 - math.exp(-max(dt, 0.0) / tau)
        return current + (target - current) * k

    def _update_label(self):
        # Hysteresis: expand/contract thresholds based on current label
        m = self.margin
        if self.label == "calm":
            t1 = self.t_calm_mod + m
            if self.level >= t1: self.label = "mod_stress"
        elif self.label == "mod_stress":
            t_down = self.t_calm_mod - m
            t_up   = self.t_mod_high + m
            if self.level <= t_down: self.label = "calm"
            elif self.level >= t_up: self.label = "stressed"
        elif self.label == "stressed":
            t_down = self.t_mod_high - m
            t_up   = self.t_high_ext + m
            if self.level <= t_down: self.label = "mod_stress"
            elif self.level >= t_up: self.label = "extreme_stress"
        elif self.label == "extreme_stress":
            t_back = self.t_high_ext - m
            if self.level <= t_back: self.label = "stressed"

    def update(self, raw_state: str, dt: float):
        target = state_to_level(raw_state)
        tau = self.tau_rise if target > self.level else self.tau_fall
        self.level = self._exp_smooth(self.level, target, dt, tau)
        self.level = max(0.0, min(1.0, self.level))
        self._update_label()
        return self.level, self.label

#------------------- main loop with eeg from file ---------------------
        
import numpy as np
import argparse
import time
from eeg_filereader import OfflineEEGFeeder, LiveArousalClassifier


def smoothstep01(x: float) -> float:
    # cubic smoothstep in [0,1]
    return x*x*(3.0 - 2.0*x)

def exp_smooth(current: float, target: float, real_dt: float, tau: float) -> float:
    """One-pole low-pass toward target with time-constant tau (seconds)."""
    if tau <= 0.0:
        return target
    k = 1.0 - math.exp(-max(real_dt, 0.0) / tau)
    return current + (target - current) * k


# Use your existing CALM/STRESSED, add two more:
PRESETS = {
    "calm": {
        "time_mul": 0.5, "freq_mul": 1.0, "contrast": 0.500,
        "phase_jitter": 1.436, "anim_speed": 0.063, "octaves": 2,
        "palette_bias": (-0.4, -0.4, 0.4), "heart_weight": 0.20,
    },
    "mod_stress": {   # a bit livelier/warmer than calm
        "time_mul": 0.5, "freq_mul": 1.0, "contrast": 0.75,
        "phase_jitter": 1.50, "anim_speed": 0.25, "octaves": 3,
        "palette_bias": (-0.10, -0.20, 0.10), "heart_weight": 0.20,
    },
    "high_stress": {  # turn it up further
        "time_mul": 0.5, "freq_mul": 1.00, "contrast": 0.80,
        "phase_jitter": 1.50, "anim_speed": 1.00, "octaves": 3,
        "palette_bias": (0.33, -0.20, -0.20), "heart_weight": 0.20,
    },
    "extreme_stress": {     # your current STRESSED
        "time_mul": 0.5, "freq_mul": 1.0, "contrast": 0.873,
        "phase_jitter": 1.469, "anim_speed": 2.000, "octaves": 4,
        "palette_bias": (0.40, -0.40, -0.40), "heart_weight": 0.20,
    },
}

KEYFRAMES = [  # s position for each preset in [0..1]
    ("calm",           0.00),
    ("mod_stress",     0.33),
    ("high_stress",    0.66),
    ("extreme_stress", 1.00),
]

def _lerp(a, b, t): return a * (1.0 - t) + b * t
def _lerp_int(a, b, t): return int(round(_lerp(float(a), float(b), t)))

def _apply_params(p):
    time_mul[None]     = p["time_mul"]
    freq_mul[None]     = p["freq_mul"]
    contrast[None]     = p["contrast"]
    phase_jitter[None] = p["phase_jitter"]
    heart_weight[None] = p["heart_weight"]
    speed[None]        = p["anim_speed"]
    octaves[None]      = int(p["octaves"])
    pr, pg, pb = p["palette_bias"]
    palette_bias[None] = ti.Vector([pr, pg, pb])

def _blend_two(a, b, t):
    out = {}
    out["time_mul"]     = _lerp(a["time_mul"],     b["time_mul"],     t)
    out["freq_mul"]     = _lerp(a["freq_mul"],     b["freq_mul"],     t)
    out["contrast"]     = _lerp(a["contrast"],     b["contrast"],     t)
    out["phase_jitter"] = _lerp(a["phase_jitter"], b["phase_jitter"], t)
    out["heart_weight"] = _lerp(a["heart_weight"], b["heart_weight"], t)
    out["anim_speed"]   = _lerp(a["anim_speed"],   b["anim_speed"],   t)
    out["octaves"]      = _lerp_int(a["octaves"],  b["octaves"],      t)
    apr, apg, apb = a["palette_bias"]; bpr, bpg, bpb = b["palette_bias"]
    out["palette_bias"] = (_lerp(apr, bpr, t), _lerp(apg, bpg, t), _lerp(apb, bpb, t))
    return out

def blend_params_keyframed(s: float):
    # Clamp s
    s = max(0.0, min(1.0, s))
    # Find bracketing keyframes
    left_name, left_s = KEYFRAMES[0]
    right_name, right_s = KEYFRAMES[-1]
    for i in range(len(KEYFRAMES) - 1):
        n0, s0 = KEYFRAMES[i]
        n1, s1 = KEYFRAMES[i + 1]
        if s >= s0 and s <= s1:
            left_name, left_s = n0, s0
            right_name, right_s = n1, s1
            break
    # Local t within the interval
    t = 0.0 if right_s == left_s else (s - left_s) / (right_s - left_s)
    pL, pR = PRESETS[left_name], PRESETS[right_name]
    p = _blend_two(pL, pR, t)
    _apply_params(p)


def norm_state(name: str) -> str:
    if not name:
        return "calm"
    return str(name).strip().lower().replace(" ", "_")

# Discrete mapping: state -> target_s in [0..1]
STATE_LEVEL = {
    "calm":                  0.00,
    "mod-stress":            0.33,
    "high-stress":           0.66,
    "extreme-stress":        1.00
}

def state_to_target(state: str) -> float:
    return STATE_LEVEL.get(norm_state(state), 0.0)



def main():
    #------- eeg file reader set up --------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg", action="append", help="Path to EEG .txt file (can repeat)", default=None)
    args = parser.parse_args()

    if args.eeg:
        EEG_FILES = args.eeg                     # from launcher (can be 1+ files)
    else:
        EEG_FILES = ["../eeg_files/fake_eeg_longblocks_calmfirst.txt"]  # fallback

    EEG_FS = 256.0
    try:
        feeder = OfflineEEGFeeder(EEG_FILES, fs=EEG_FS, chunk=32, speed=1.0, loop=True, buffer_s=8.0)
        clf    = LiveArousalClassifier(fs=EEG_FS, lf=(4,12), hf=(13,40), win_s=4.0)
    except Exception as e:
        print("EEG feeder disabled:", e)
        feeder = None 
        clf = None


    window = ti.ui.Window("Mandala (GUI)", (W, H))
    canvas = window.get_canvas()
    gui = window.get_gui()

    # target (user) & smoothed state
    target_s = 0.0     # 0 = CALM, 1 = STRESSED
    smooth_s = 0.0

    # init fields
    blend_params_keyframed(smooth_s)
    smoother = SoftStateSmoother(tau_rise=0.25, tau_fall=0.80, margin=0.08) 

    t  = 0.0
    dt = 1.0 / 60.0 
    last_time = time.perf_counter()

    while window.running:

        feeder.step_once()
        state, ratio, _ = clf.update(feeder.get_buffer())
        
        now       = time.perf_counter()
        real_dt   = now - last_time
        last_time = now

        # Use classifier 'state' only (ignore ratio for driving)
        level, stable_state = smoother.update(state, real_dt)# smoothed state
        raw_target   = state_to_target(stable_state)       # discrete -> scalar in [0..1]
        # Smooth the discrete target for nice transitions
        #tau      = tau_rise if raw_target > smooth_s else tau_fall
        smooth_s = level
        # Push blended parameters
        blend_params_keyframed(smooth_s)

        #print("state:", state)
        #print("stable state", stable_state)

        with gui.sub_window("Readout", 0.70, 0.02, 0.27, 0.20):
            gui.text(f"Raw state: {state}")
            gui.text(f"Stable state: {stable_state}")
            if ratio == ratio:
                gui.text(f"HF/LF: {ratio:.3f}")
            gui.text(f"Smoothed level: {smooth_s:.3f}")

        #--------------render---------------------
        render(t)
        canvas.set_image(img)
        window.show()
        t += dt * speed[None]

if __name__ == "__main__":
    main()

