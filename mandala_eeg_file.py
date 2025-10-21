# heart_plasma_taichi.py


"""
this CODE does:
1) takes the hf/lf ratio to determine stress
2) Internally, computes a smoothed scalar smooth_s using
exponential low-pass with asymmetric time constants 
(tau_rise=0.25, tau_fall=0.60).
3) for every frame, blends CALM→STRESSED for:
                time multiplier, frequency, contrast, phase jitter, 
                animation speed, octaves, heart SDF weight, and palette bias (RGB).
4) The animation time uses the blended anim_speed, 
so even the tempo eases smoothly.


Possible tweak ideas:
-> To make any single parameter react faster/slower than the rest, give it its own tau and run a separate exp_smooth for that 
control instead of blending all from a common smooth_s.
-> to add a little “breathing” motion even when target is steady, 
add a tiny low-frequency wobble to target_s before smoothing (e.g., target_s + 0.03*sin(0.2*t) and clamp).


"""


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

def lerp(a: float, b: float, t: float) -> float:
    return a * (1.0 - t) + b * t

def lerp_int(a: int, b: int, t: float) -> int:
    return int(round(lerp(float(a), float(b), t)))

CALM = {
    "time_mul":     0.406,
    "freq_mul":     0.500,
    "contrast":     0.500,
    "phase_jitter": 1.436,
    "anim_speed":   0.063,
    "octaves":      2,
    "palette_bias": (-0.4, -0.4, 0.4),
    "heart_weight": 0.12,   # modest, coherent warp
}

# Choose stressed values to feel sharper, faster, warmer
STRESSED = {
    "time_mul":     1.467,
    "freq_mul":     1.041,
    "contrast":     0.873,
    "phase_jitter": 1.469,
    "anim_speed":   2.000,
    "octaves":      4,
    "palette_bias": (0.40, -0.40, -0.40),
    "heart_weight": 0.260,
}
def blend_params(s: float):
    """Blend CALM→STRESSED using scalar s in [0,1]."""
    time_mul[None]     = lerp(CALM["time_mul"],     STRESSED["time_mul"],     s)
    freq_mul[None]     = lerp(CALM["freq_mul"],     STRESSED["freq_mul"],     s)
    contrast[None]     = lerp(CALM["contrast"],     STRESSED["contrast"],     s)
    phase_jitter[None] = lerp(CALM["phase_jitter"], STRESSED["phase_jitter"], s)
    heart_weight[None] = lerp(CALM["heart_weight"], STRESSED["heart_weight"], s)
    speed[None]        = lerp(CALM["anim_speed"],   STRESSED["anim_speed"],   s)
    octaves[None]      = lerp_int(CALM["octaves"],  STRESSED["octaves"],      s)

    pr, pg, pb = CALM["palette_bias"]
    qr, qg, qb = STRESSED["palette_bias"]
    palette_bias[None] = ti.Vector([lerp(pr, qr, s),
                                    lerp(pg, qg, s),
                                    lerp(pb, qb, s)])

def map_ratio_to_target(ratio: float,
                        r_min: float = 4.0,   # calm boundary  (HF/LF small)
                        r_max: float = 9.0    # stressed bound (HF/LF large)
                        ) -> float:
    """
    Map classifier ratio -> [0..1] target intensity.
    Log-space normalization makes it perceptual; smoothstep eases the ends.
    """
    if ratio is None or not np.isfinite(ratio) or ratio <= 0:
        return 0.0  # treat bad values as calm

    x = (np.log(ratio) - np.log(r_min)) / (np.log(r_max) - np.log(r_min))
    x = float(np.clip(x, 0.0, 1.0))
    return smoothstep01(x)




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

    tau_rise = 0.25
    tau_fall = 0.60

    # init fields
    blend_params(smooth_s)

    t  = 0.0
    dt = 1.0 / 60.0 
    last_time = time.perf_counter()

    while window.running:

        feeder.step_once()
        state, ratio, _ = clf.update(feeder.get_buffer())
        
        now       = time.perf_counter()
        real_dt   = now - last_time
        last_time = now

        raw_target = map_ratio_to_target(ratio, r_min=0.1, r_max=9.0)
        tau        = tau_rise if raw_target > smooth_s else tau_fall
        smooth_s   = exp_smooth(smooth_s, raw_target, real_dt, tau)
        
        blend_params(smooth_s)

        with gui.sub_window("Readout", 0.70, 0.02, 0.27, 0.16):
            gui.text(f"State: {state}  |  HF/LF: {ratio:.3f}" if ratio == ratio else f"State: {state}")
            gui.text(f"Target(raw): {raw_target:.3f}  |  Smoothed: {smooth_s:.3f}")

        #--------------render---------------------
        render(t)
        canvas.set_image(img)
        window.show()
        t += dt * speed[None]

if __name__ == "__main__":
    main()

