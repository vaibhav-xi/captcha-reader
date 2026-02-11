import os
import random
import string
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "../dataset/targeted_images_v3"
COUNT = 15000

W = 200
H = 50

FONT_DIR = "fonts"
FONTS = [os.path.join(FONT_DIR,f) for f in os.listdir(FONT_DIR)]

CHARS = string.ascii_letters + string.digits + "@=#"

os.makedirs(OUT_DIR, exist_ok=True)

def label_lI_collision():
    pool = "lIi"
    return "".join(random.choice(pool if random.random()<0.6 else CHARS) for _ in range(6))

def label_double():
    s = [random.choice(CHARS) for _ in range(5)]
    k = random.randint(0,4)
    s.insert(k, s[k])
    return "".join(s)

def label_qg():
    pool = "qg"
    return "".join(random.choice(pool if random.random()<0.5 else CHARS) for _ in range(6))

def label_OQ():
    pool = "OQ0"
    return "".join(random.choice(pool if random.random()<0.5 else CHARS) for _ in range(6))

def label_thin():
    pool = "ilItfjr1"
    return "".join(random.choice(pool if random.random()<0.5 else CHARS) for _ in range(6))

def label_anti_y():
    pool = "vyY"
    return "".join(random.choice(pool if random.random()<0.5 else CHARS) for _ in range(6))

def glyph_ok(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black = np.mean(gray < 40)

    if black < 0.01:
        return False

    if black > 0.40:
        return False

    return True

def orange_bg():
    base = np.full((H,W,3), (60,170,240), np.uint8)
    noise = np.random.normal(0,12,(H,W,3))
    return np.clip(base + noise,0,255).astype(np.uint8)

def draw_text_safe(label, thin=False, border_touch=False):

    for _ in range(8):

        img = orange_bg()
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)

        font_path = random.choice(FONTS)
        size = random.randint(34,42)
        font = ImageFont.truetype(font_path, size)

        x = random.randint(6,16)
        y = random.randint(2,8)

        if border_touch:
            x = random.randint(-3,4)

        for ch in label:

            if ch in "@=#":
                stroke = 1
            else:
                stroke = 1 if thin else random.choice([1,2])

            d.text(
                (x,y),
                ch,
                font=font,
                fill=(0,0,0),
                stroke_width=stroke,
                stroke_fill=(0,0,0)
            )

            x += size - random.randint(7,11)

        arr = np.array(pil)

        if glyph_ok(arr):
            return arr

    return arr

def add_green_edge_noise(img):
    g = img.copy()
    gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,40,120)

    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return g

    k = int(len(xs)*0.025)

    for i in np.random.choice(len(xs), k, replace=False):
        y,x = ys[i], xs[i]
        g[y,x] = [0, random.randint(110,180), 0]

    return g

def thin_erode(img):
    if random.random() < 0.5:
        k = np.ones((2,2),np.uint8)
        img = cv2.erode(img,k,iterations=1)
    return img

CATS = [
    ("lI", 0.14, label_lI_collision),
    ("dbl",0.10, label_double),
    ("qg", 0.10, label_qg),
    ("OQ", 0.08, label_OQ),
    ("thin",0.08, label_thin),
    ("border",0.07, lambda: "".join(random.choice(CHARS) for _ in range(6))),
    ("anti_y",0.07, label_anti_y),
]

def pick_cat():
    r = random.random()
    s = 0
    for name,p,fn in CATS:
        s += p
        if r <= s:
            return name, fn
    return CATS[-1][0], CATS[-1][2]

for i in range(COUNT):

    cat, fn = pick_cat()
    label = fn()

    thin = (cat == "thin")
    border = (cat == "border")

    img = draw_text_safe(label, thin, border)
    img = add_green_edge_noise(img)

    if thin:
        img = thin_erode(img)

    if random.random() < 0.4:
        img = cv2.GaussianBlur(img,(3,3),0)

    cv2.imwrite(os.path.join(OUT_DIR,f"{label}.png"), img)

print("Done →", COUNT)
