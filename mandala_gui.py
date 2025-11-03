# heart_plasma_taichi.py
import taichi as ti
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

@ti.func #returns a signed distance shaped like a heart
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



def main():
    window = ti.ui.Window("Mandala (GUI)", (W, H))
    canvas = window.get_canvas()
    gui = window.get_gui()

    t = 0.0
    dt = 1.0 / 60.0  # animation speed similar to iTime

    while window.running:
        # ---------- GUI Controls ----------
        with gui.sub_window("GUI", 0.02, 0.02, 0.30, 0.20):
            gui.text("Manual Controls (0=calm ... higher=more intense)")
            # returns current value on most Taichi builds
            time_mul[None]     = gui.slider_float("Time mul",        time_mul[None],     0.05, 3.0 )
            freq_mul[None]     = gui.slider_float("Frequency mul",   freq_mul[None],     0.5,  3.0 )
            contrast[None]     = gui.slider_float("Contrast",        contrast[None],     0.5,  2.5)
            heart_weight[None] = gui.slider_float("Heart weight",    heart_weight[None], 0.0,  0.6)
            phase_jitter[None] = gui.slider_float("Phase jitter",    phase_jitter[None], 0.0,  2.0)
            speed[None]        = gui.slider_float("Animation speed", speed[None],        0.05, 2.0)

            # octaves as int slider (2..4 works well for perf/clarity)
            oct = int(round(gui.slider_float("Octaves (2-4)",float(octaves[None]), 2.0, 4.0)))
            octaves[None] = max(1, min(4, oct))

            gui.text("Palette bias (RGB): negative=bluer, positive=redder/warmer")
            r = gui.slider_float("Bias R", float(palette_bias[None][0]), -0.4, 0.4)
            g = gui.slider_float("Bias G", float(palette_bias[None][1]), -0.4, 0.4)
            b = gui.slider_float("Bias B", float(palette_bias[None][2]), -0.4, 0.4)
            palette_bias[None] = ti.Vector([r, g, b])

        #--------------render---------------------
        render(t)
        canvas.set_image(img)
        window.show()
        t += dt * speed[None]

if __name__ == "__main__":
    main()

