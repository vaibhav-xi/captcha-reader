import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

OUTPUT_DIR = "../../dataset/confusion_boost_v1"
TOTAL_IMAGES = 10000

WIDTH = 265
HEIGHT = 67

ORANGE = (255,165,0)
BLACK  = (0,0,0)

CHARS = string.ascii_letters + string.digits + "@#="

CONFUSION_GROUPS = [
    "lIiJj",
    "gq",
    "OQ0",
    "vy",
    "mn",
    "EF",
    "B8",
    "S5",
    "Z2",
]

ALL_CONF = "".join(CONFUSION_GROUPS)

def confusion_pair_text():
    """Dense confusion-only string"""
    g = random.choice(CONFUSION_GROUPS)
    L = random.randint(4,5)
    return "".join(random.choice(g) for _ in range(L))


def mixed_confusion_text():
    """Mostly confusion chars with noise"""
    L = random.randint(4,6)
    t = []
    for _ in range(L):
        if random.random() < 0.7:
            g = random.choice(CONFUSION_GROUPS)
            t.append(random.choice(g))
        else:
            t.append(random.choice(CHARS))
    return "".join(t)


def alternating_confusion():
    """Pattern like lIlI gqgq"""
    g = random.choice(CONFUSION_GROUPS)
    if len(g) < 2:
        return confusion_pair_text()
    a,b = random.sample(g,2)
    L = random.randint(4,6)
    return "".join(a if i%2==0 else b for i in range(L))


def double_letter_confusion():
    """Stress double-letter failures"""
    base = mixed_confusion_text()
    pos = random.randrange(len(base))
    return (base[:pos] + base[pos] + base[pos:])[:6]


def normal_text():
    return "".join(random.choice(CHARS) for _ in range(random.randint(4,6)))


def generate_text():
    r = random.random()

    if r < 0.40: return confusion_pair_text()
    if r < 0.65: return mixed_confusion_text()
    if r < 0.80: return alternating_confusion()
    if r < 0.92: return double_letter_confusion()
    return normal_text()

def load_font(size=50):
    for f in [
        "../fonts/DejaVuSans.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    ]:
        try:
            return ImageFont.truetype(f,size)
        except:
            pass
    return ImageFont.load_default()


def jitter_pts(pts,j=10):
    return [(x+random.randint(-j,j), y+random.randint(-j,j)) for x,y in pts]


def dominant_trapezoid():
    y_top = random.randint(18,42)
    y_bot = random.randint(HEIGHT-8,HEIGHT+18)

    left  = random.randint(-70,25)
    right = random.randint(WIDTH-45,WIDTH+70)

    skew_top = random.randint(-30,25)
    skew_bot = random.randint(-25,35)

    pts = [
        (left,y_top),
        (right,y_top+skew_top),
        (right,y_bot+skew_bot),
        (left,y_bot)
    ]

    return jitter_pts(pts,12)


def draw_overlay(mask_draw):
    mask_draw.polygon(dominant_trapezoid(), fill=255)

    if random.random() < 0.45:
        tri = [
            (-40,random.randint(0,HEIGHT)),
            (WIDTH+40,random.randint(-10,HEIGHT)),
            (WIDTH+40,random.randint(HEIGHT//2,HEIGHT+40))
        ]
        mask_draw.polygon(jitter_pts(tri,8), fill=255)


def draw_text_mask(mask,text,font):

    d = ImageDraw.Draw(mask)

    boxes = [d.textbbox((0,0),c,font=font) for c in text]
    ws = [b[2]-b[0] for b in boxes]
    hs = [b[3]-b[1] for b in boxes]

    total_w = sum(ws)
    max_h = max(hs)

    x = (WIDTH-total_w)//2 + random.randint(-6,6)
    y0 = (HEIGHT-max_h)//2 + random.randint(-4,4)

    for c,w in zip(text,ws):
        y = y0 + random.randint(-8,8)
        d.text((x,y),c,fill=255,font=font)
        x += w + random.randint(-3,3)

def thin_stroke(mask):
    if random.random() < 0.55:
        return mask.filter(ImageFilter.MinFilter(3))
    return mask


def heavy_edge_tint(img,mask):
    arr = np.array(img).astype(np.int16)
    m = np.array(mask)

    eroded = Image.fromarray(m).filter(ImageFilter.MinFilter(3))
    edge = m - np.array(eroded)

    ys,xs = np.where(edge>0)

    for y,x in zip(ys,xs):
        if random.random()<0.6:
            arr[y,x,0]+=random.randint(0,25)
            arr[y,x,1]+=random.randint(20,45)
            arr[y,x,2]-=random.randint(0,20)

    return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))


def squeeze(img):
    if random.random()<0.35:
        w,h = img.size
        img2 = img.resize((int(w*0.90),h))
        canvas = Image.new("RGB",(w,h),ORANGE)
        canvas.paste(img2,((w-img2.size[0])//2,0))
        return canvas
    return img

def generate_one(text,font,path):

    img = Image.new("RGB",(WIDTH,HEIGHT),ORANGE)

    overlay = Image.new("L",(WIDTH,HEIGHT),0)
    draw_overlay(ImageDraw.Draw(overlay))
    overlay = overlay.filter(ImageFilter.GaussianBlur(1.2))
    img.paste(BLACK,mask=overlay)

    text_mask = Image.new("L",(WIDTH,HEIGHT),0)
    draw_text_mask(text_mask,text,font)

    orig_mask = text_mask.copy()
    text_mask = thin_stroke(text_mask)
    text_mask = text_mask.filter(ImageFilter.GaussianBlur(0.6))

    orange_mask = ImageChops.multiply(text_mask,overlay)
    black_mask  = ImageChops.subtract(text_mask,orange_mask)

    img.paste(BLACK, mask=black_mask)
    img.paste(ORANGE,mask=orange_mask)

    img = heavy_edge_tint(img,orig_mask)

    if random.random()<0.55:
        img = img.filter(ImageFilter.GaussianBlur(0.4))

    img = squeeze(img)

    img.save(path,quality=random.randint(82,92))

def main():

    os.makedirs(OUTPUT_DIR,exist_ok=True)
    font = load_font(50)

    made = 0
    used = set()

    while made < TOTAL_IMAGES:

        t = generate_text()

        if t in used:
            continue

        used.add(t)

        p = os.path.join(OUTPUT_DIR,f"{t}.png")
        generate_one(t,font,p)

        made += 1
        if made % 200 == 0:
            print("Generated:",made)

    print("\nDone:",made)


if __name__ == "__main__":
    main()
