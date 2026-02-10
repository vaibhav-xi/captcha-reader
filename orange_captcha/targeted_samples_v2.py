import os
import random
import string
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = "../dataset/targeted_images_v2"
COUNT = 15000

W = 200
H = 50

FONT_DIR = "fonts"
FONTS = [os.path.join(FONT_DIR,f) for f in os.listdir(FONT_DIR)]

CHARS = string.ascii_letters + string.digits + "@=#"

os.makedirs(OUT_DIR, exist_ok=True)

def label_lI_collision():
    pool = "lIi"
    return "".join(random.choice(pool if random.random()<0.6 else CHARS)
                   for _ in range(6))

def label_double():
    s = [random.choice(CHARS) for _ in range(5)]
    k = random.randint(0,4)
    s.insert(k, s[k])
    return "".join(s)

def label_qg():
    pool = "qg"
    return "".join(random.choice(pool if random.random()<0.5 else CHARS)
                   for _ in range(6))

def label_OQ():
    pool = "OQ0"
    return "".join(random.choice(pool if random.random()<0.5 else CHARS)
                   for _ in range(6))

def label_thin():
    pool = "ilItfjr1"
    return "".join(random.choice(pool if random.random()<0.5 else CHARS)
                   for _ in range(6))

def label_anti_y():
    pool = "vyY"
    return "".join(random.choice(pool if random.random()<0.5 else CHARS)
                   for _ in range(6))

def draw_text(label, thin=False, border_touch=False):
    img = Image.new("RGB", (W,H), (255,255,255))
    d = ImageDraw.Draw(img)

    font_path = random.choice(FONTS)
    size = random.randint(34,40)
    font = ImageFont.truetype(font_path, size)

    x = random.randint(5,15)
    y = random.randint(2,8)

    if border_touch:
        x = random.randint(-2,4)

    for ch in label:

        stroke = 1 if thin else random.randint(2,3)

        d.text(
            (x,y),
            ch,
            font=font,
            fill=(0,0,0),
            stroke_width=stroke,
            stroke_fill=(0,0,0)
        )

        x += size - random.randint(6,10)

    return np.array(img)

def add_edge_green_noise(img):

    g = img.copy()
    gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,40,120)

    ys, xs = np.where(edges > 0)
    k = int(len(xs)*0.03)

    for i in np.random.choice(len(xs), k, replace=False):
        y,x = ys[i], xs[i]
        g[y,x] = [0, random.randint(120,180), 0]

    return g

def q_tail_variants(img,label):

    if "q" not in label and "Q" not in label:
        return img

    if random.random() < 0.5:
        h,w,_ = img.shape
        cv2.line(img,
                 (w-30, h-8),
                 (w-18, h-2),
                 (0,0,0),
                 random.randint(1,2))
    return img

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

    img = draw_text(label, thin=thin, border_touch=border)

    img = add_edge_green_noise(img)
    img = q_tail_variants(img,label)

    if thin:
        img = thin_erode(img)

    # slight blur like real samples
    if random.random() < 0.4:
        img = cv2.GaussianBlur(img,(3,3),0)

    path = os.path.join(OUT_DIR, f"{label}_{i}.png")
    cv2.imwrite(path, img)

print("Done →", COUNT)
