import os
import cv2
import numpy as np
import random
import string

OUT_DIR = "../dataset/targeted_images"
N_SAMPLES = 15000

W = 200
H = 50

CHARS = string.ascii_letters + string.digits + "@=#"

os.makedirs(OUT_DIR, exist_ok=True)

FONTS = [
    "fonts/DejaVuSans.ttf",
]

def label_lI_collision():
    pool = "lI1i"
    base = [random.choice(CHARS) for _ in range(6)]
    pos = random.randint(1,4)
    base[pos] = random.choice(pool)
    return "".join(base)

def label_double():
    base = [random.choice(CHARS) for _ in range(6)]
    pos = random.randint(1,4)
    c = random.choice(CHARS)
    base[pos] = c
    base[pos+1] = c
    return "".join(base)

def label_normal():
    L = random.randint(5,6)
    return "".join(random.choice(CHARS) for _ in range(L))

def render_text(label, thin=False, border_touch=False):

    img = np.full((H,W,3), 255, np.uint8)

    font = random.choice(FONTS)
    font_size = random.randint(28,34)

    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    fnt = ImageFont.truetype(font, font_size)

    tw, th = draw.textbbox((0,0), label, font=fnt)[2:]

    if border_touch:
        x = random.randint(-5, 5)
        y = random.randint(5, H-th-2)
    else:
        x = (W - tw)//2 + random.randint(-5,5)
        y = (H - th)//2 + random.randint(-3,3)

    draw.text((x,y), label, font=fnt, fill=(0,0,0))

    img = np.array(pil)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if thin:
        gray = cv2.erode(gray, np.ones((2,2),np.uint8), iterations=1)

    if random.random() < 0.6:
        gray = cv2.GaussianBlur(gray,(3,3),0)

    noise = np.random.normal(0,8,gray.shape).astype(np.int16)
    gray = np.clip(gray.astype(np.int16)+noise,0,255).astype(np.uint8)

    return gray

def sample_category():
    r = random.random()

    if r < 0.08:
        return "lI"
    elif r < 0.16:
        return "double"
    elif r < 0.23:
        return "border"
    else:
        return "thin"

for i in range(N_SAMPLES):

    cat = sample_category()

    if cat == "lI":
        label = label_lI_collision()
        img = render_text(label)

    elif cat == "double":
        label = label_double()
        img = render_text(label)

    elif cat == "border":
        label = label_normal()
        img = render_text(label, border_touch=True)

    else:
        label = label_normal()
        img = render_text(label, thin=True)

    cv2.imwrite(f"{OUT_DIR}/{label}.png", img)

    if i % 500 == 0:
        print(i, "generated")

print("\nDone — targeted dataset ready.")
