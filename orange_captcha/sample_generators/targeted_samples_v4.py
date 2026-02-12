import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

OUTPUT_DIR = "/Volumes/samsung_980/projects/captcha-reader/dataset/generated_samples_v15"
TOTAL_IMAGES = 12000

EXISTING_DIRS = [
    "../../dataset/generated_samples_v2",
    "../../dataset/generated_samples_v3",
    "../../dataset/generated_samples_v4",
    "../../dataset/generated_samples_v5",
    "../../dataset/generated_samples_v6",
    "../../dataset/generated_samples_v7",
    "../../dataset/generated_samples_v8",
    "../../dataset/generated_samples_v9",
    "../../dataset/generated_samples_v10",
    "../../dataset/generated_samples_v11",
    "../../dataset/generated_samples_v12",
    "../../dataset/generated_samples_v13",
    "../../dataset/generated_samples_v14",
    "../../dataset/targeted_images",
    "../../dataset/targeted_images_v2",
    "../../dataset/targeted_images_v3",
    "../../dataset/targeted_images_v4",
    "../../dataset/targeted_images_v5",
    "../../dataset/targeted_images_v6",
]

WIDTH = 265
HEIGHT = 67

ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)

CHARS = string.ascii_letters + string.digits + "@#="

CONFUSION_GROUPS = [
    "lI1Jj",
    "O0Q",
    "B8",
    "S5",
    "Z2",
    "G6qg",
    "m n".replace(" ",""),
    "v y".replace(" ",""),
]

CONFUSION_MAP = {}
for g in CONFUSION_GROUPS:
    for c in g:
        CONFUSION_MAP[c] = list(g)

def random_text():
    """Baseline random"""
    return "".join(random.choice(CHARS) for _ in range(random.randint(4, 6)))

def double_letter_text():
    t = list(random_text())
    pos = random.randrange(len(t))
    t.insert(pos, t[pos])
    return "".join(t[:6])

def confusion_text():
    L = random.randint(4,6)
    t = [random.choice(CHARS) for _ in range(L)]

    k = random.randint(2, min(3, L))
    idxs = random.sample(range(L), k)

    for pos in idxs:
        grp = random.choice(CONFUSION_GROUPS)
        t[pos] = random.choice(grp)

    return "".join(t)

def case_flip_text():
    """Mixed case stress"""
    t = random_text()
    out = []
    for c in t:
        if c.isalpha() and random.random() < 0.5:
            c = c.swapcase()
        out.append(c)
    return "".join(out)


def swapped_variant_text():
    """Take random base and swap confusable chars"""
    t = list(random_text())

    for i,c in enumerate(t):
        if c in CONFUSION_MAP and random.random() < 0.6:
            t[i] = random.choice(CONFUSION_MAP[c])

    return "".join(t)


def generate_targeted_text():
    r = random.random()

    if r < 0.50:
        return confusion_text()
    elif r < 0.72:
        return swapped_variant_text()
    elif r < 0.88:
        return double_letter_text()
    elif r < 0.96:
        return case_flip_text()
    else:
        return random_text()
    
def thin_stroke_variant(text_mask):
    if random.random() < 0.35:
        return text_mask.filter(ImageFilter.MinFilter(3))
    return text_mask

def bowl_stress(img):
    if random.random() < 0.30:
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120))
    return img

def squeeze_variant(img):
    if random.random() < 0.25:
        w,h = img.size
        img = img.resize((int(w*0.92), h))
        canvas = Image.new("RGB",(w,h),ORANGE)
        canvas.paste(img, ((w-img.size[0])//2,0))
        return canvas
    return img

def load_font(size=50):
    for f in [
        "../fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(f, size)
        except:
            continue
    return ImageFont.load_default()

def jitter_pts(pts, j=10):
    return [(x+random.randint(-j,j), y+random.randint(-j,j)) for x,y in pts]

def dominant_trapezoid():
    y_top = random.randint(18, 42)
    y_bot = random.randint(HEIGHT - 8, HEIGHT + 18)

    left  = random.randint(-70, 25)
    right = random.randint(WIDTH - 45, WIDTH + 70)

    skew_top = random.randint(-30, 25)
    skew_bot = random.randint(-25, 35)

    pts = [
        (left, y_top),
        (right, y_top + skew_top),
        (right, y_bot + skew_bot),
        (left, y_bot)
    ]

    return jitter_pts(pts, 12)

def draw_real_overlay(mask_draw):
    mask_draw.polygon(dominant_trapezoid(), fill=255)

    if random.random() < 0.4:
        tri = [
            (-40, random.randint(0,HEIGHT)),
            (WIDTH+40, random.randint(-10,HEIGHT)),
            (WIDTH+40, random.randint(HEIGHT//2,HEIGHT+40))
        ]
        mask_draw.polygon(jitter_pts(tri,8), fill=255)

def draw_text_mask(mask, text, font):

    d = ImageDraw.Draw(mask)

    boxes = [d.textbbox((0,0),c,font=font) for c in text]
    ws = [b[2]-b[0] for b in boxes]
    hs = [b[3]-b[1] for b in boxes]

    total_w = sum(ws)
    max_h = max(hs)

    x = (WIDTH-total_w)//2 + random.randint(-6,6)
    y0 = (HEIGHT-max_h)//2 + random.randint(-3,3)

    for c,w in zip(text,ws):
        y = y0 + random.randint(-7,7)
        d.text((x,y),c,fill=255,font=font)
        x += w + random.randint(-3,3)

def edge_tint(img, text_mask):
    arr = np.array(img).astype(np.int16)
    m = np.array(text_mask)

    eroded = Image.fromarray(m).filter(ImageFilter.MinFilter(3))
    edge = m - np.array(eroded)

    ys, xs = np.where(edge > 0)

    for y,x in zip(ys, xs):
        if random.random() < 0.35:
            arr[y,x,0] += random.randint(0,20)
            arr[y,x,1] += random.randint(15,35)
            arr[y,x,2] -= random.randint(0,15)

    arr = np.clip(arr,0,255).astype(np.uint8)
    return Image.fromarray(arr)

def generate_one(text, font, save_path):

    img = Image.new("RGB",(WIDTH,HEIGHT),ORANGE)

    overlay_mask = Image.new("L",(WIDTH,HEIGHT),0)
    draw_real_overlay(ImageDraw.Draw(overlay_mask))
    overlay_mask = overlay_mask.filter(ImageFilter.GaussianBlur(1.1))
    img.paste(BLACK, mask=overlay_mask)

    text_mask = Image.new("L",(WIDTH,HEIGHT),0)
    draw_text_mask(text_mask, text, font)
    
    orig_text_mask = text_mask.copy()
    text_mask = thin_stroke_variant(text_mask)
    
    text_mask = text_mask.filter(ImageFilter.GaussianBlur(0.5))

    orange_text_mask = ImageChops.multiply(text_mask, overlay_mask)
    black_text_mask  = ImageChops.subtract(text_mask, orange_text_mask)

    img.paste(BLACK, mask=black_text_mask)
    img.paste(ORANGE, mask=orange_text_mask)

    if random.random() < 0.75:
        img = edge_tint(img, orig_text_mask)

    if random.random() < 0.45:
        img = img.filter(ImageFilter.GaussianBlur(0.35))
        
    img = bowl_stress(img)
    img = squeeze_variant(img)
    img.save(save_path, quality=random.randint(82,92))

def load_existing_labels():
    used=set()
    for d in EXISTING_DIRS:
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            if f.lower().endswith(".png"):
                used.add(os.path.splitext(f)[0])
    return used

def main():

    os.makedirs(OUTPUT_DIR,exist_ok=True)
    font = load_font(50)

    used = load_existing_labels()
    print("Existing labels:",len(used))

    made = 0

    while made < TOTAL_IMAGES:

        t = generate_targeted_text()

        if t in used:
            continue

        used.add(t)

        path = os.path.join(OUTPUT_DIR,f"{t}.png")
        generate_one(t,font,path)

        made += 1
        if made % 100 == 0:
            print("Generated:",made)

    print("\nDone —",made,"new targeted samples")

if __name__ == "__main__":
    main()
