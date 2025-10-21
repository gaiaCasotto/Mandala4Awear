# heart_plasma_taichi.py
import taichi as ti

ti.init(arch=ti.gpu)  # or ti.cpu

# --- Config ---
W, H = 800, 800

# --- Buffers ---
img = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))

vec2 = ti.types.vector(2, ti.f32)
vec3 = ti.types.vector(3, ti.f32)

# --- Helpers (GLSL-like) ---
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
    res = a + b * ti.cos(two_pi * (c * t + d))
    return res

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

@ti.kernel
def render(iTime: ti.f32):
    for x, y in img:
        # fragCoord in [0, W/H)
        frag = vec2(ti.cast(x, ti.f32), ti.cast(y, ti.f32))

        iRes = vec2(float(W), float(H))
        # uv like Shadertoy: (-aspect/2..aspect/2, -0.5..0.5) with division by height
        uv = (frag * 2.0 - iRes) / iRes.y
        uv0 = uv
        finalColor = vec3(0.0, 0.0, 0.0)

        for i in range(4):
            fi = ti.cast(i, ti.f32)

            uvt = fract2(uv * 1.5) - vec2(0.5, 0.5)

            d = length2(uvt) * ti.exp(-length2(uv0))

            # Distance mod by heart SDF
            d += sdHeart(uvt) * 0.1

            col = palette(length2(uv0) + fi * 0.4 + iTime * 0.4)

            d = ti.sin(d * 8.0 + iTime) / 8.0
            d = ti.abs(d)

            # avoid div-by-zero blowups; keep the shader feel
            eps = 1e-4
            d = ti.pow(0.01 / ti.max(d, eps), 1.2)

            finalColor += col * d

            # for the next octave (match GLSL’s in-loop uv reuse)
            uv = uvt

        # Optional tonemap/clamp to keep it pretty
        finalColor = finalColor / (1.0 + finalColor)  # simple Reinhard
        img[x, y] = ti.min(finalColor, vec3(1.0, 1.0, 1.0))



def main():
    window = ti.ui.Window("Heart Plasma (Taichi)", (W, H))
    canvas = window.get_canvas()

    t = 0.0
    speed = 0.25
    dt = 1.0 / 240.0  # animation speed similar to iTime
    while window.running:
        render(t)
        canvas.set_image(img)
        window.show()
        t += dt

if __name__ == "__main__":
    main()

