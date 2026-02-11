import os
import random
import string
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "../../dataset/targeted_samples_v5"
COUNT = 500

W = 200
H = 50

FONT_DIR = "../fonts"
FONTS = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR)]

CHARS = string.ascii_letters + string.digits + "@=#"

os.makedirs(OUT_DIR, exist_ok=True)

CONF_lI = "lIi"
CONF_qg = "qg"
CONF_OQ = "OQ0"
CONF_yv = "yYv"

THIN_SET = "ilItfjr1"

def base_label():
    return [random.choice(CHARS) for _ in range(6)]

def inject_confusion(pool, strength=2):
    s = base_label()
    for _ in range(strength):
        k = random.randint(0, 5)
        s[k] = random.choice(pool)
    return "".join(s)

def label_lI():
    return inject_confusion(CONF_lI, 3)

def label_qg():
    return inject_confusion(CONF_qg, 2)

def label_OQ():
    return inject_confusion(CONF_OQ, 2)

def label_yv():
    return inject_confusion(CONF_yv, 2)

def label_thin():
    return inject_confusion(THIN_SET, 3)

def label_double():
    s = base_label()
    k = random.randint(1, 4)
    s[k] = s[k-1]
    return "".join(s)

def label_border():
    return "".join(base_label())

CATS = [
    ("lI",    0.22, label_lI),
    ("qg",    0.16, label_qg),
    ("OQ",    0.14, label_OQ),
    ("yv",    0.14, label_yv),
    ("thin",  0.14, label_thin),
    ("double",0.12, label_double),
    ("border",0.08, label_border),
]

def pick_cat():
    r = random.random()
    s = 0
    for name, p, fn in CATS:
        s += p
        if r <= s:
            return name, fn
    return CATS[-1][0], CATS[-1][2]

def orange_bg():
    base = np.full((H, W, 3), (60, 170, 240), np.uint8)
    noise = np.random.normal(0, 10, (H, W, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)

def glyph_ok(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black = np.mean(gray < 40)
    return 0.01 < black < 0.38

def draw_text_safe(label, thin=False, border_touch=False):

    for _ in range(8):

        img = orange_bg()
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)

        font_path = random.choice(FONTS)
        size = random.randint(34, 42)
        font = ImageFont.truetype(font_path, size)

        x = random.randint(8, 18)
        y = random.randint(2, 8)

        if border_touch:
            x = random.randint(-2, 4)

        for ch in label:

            if ch in "@=#":
                stroke = 1
            else:
                stroke = 1 if thin else random.choice([1, 2])

            d.text(
                (x, y),
                ch,
                font=font,
                fill=(0, 0, 0),
                stroke_width=stroke,
                stroke_fill=(0, 0, 0)
            )

            x += size - random.randint(7, 11)

        arr = np.array(pil)

        if glyph_ok(arr):
            return arr

    return arr

def add_green_edge_noise(img):
    if random.random() > 0.6:
        return img

    g = img.copy()
    gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return g

    k = int(len(xs) * 0.015)

    for i in np.random.choice(len(xs), k, replace=False):
        y, x = ys[i], xs[i]
        g[y, x] = [0, random.randint(110, 170), 0]

    return g

def thin_erode(img):
    if random.random() < 0.25:
        k = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, k, iterations=1)
    return img

seen = {}

def unique_name(label):
    if label not in seen:
        seen[label] = 0
    seen[label] += 1
    return f"{label}"

for i in range(COUNT):

    cat, fn = pick_cat()
    label = fn()

    thin = (cat == "thin")
    border = (cat == "border")

    img = draw_text_safe(label, thin, border)
    img = add_green_edge_noise(img)

    if thin:
        img = thin_erode(img)

    if random.random() < 0.35:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    name = unique_name(label)
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}.png"), img)

print("Done →", COUNT)
