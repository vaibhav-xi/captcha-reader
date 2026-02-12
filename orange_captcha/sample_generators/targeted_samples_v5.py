import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

OUTPUT_DIR = "/Volumes/samsung_980/projects/captcha-reader/dataset/generated_samples_v12"
TOTAL_IMAGES = 30000

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
    "../../dataset/targeted_images",
    "../../dataset/targeted_images_v2",
    "../../dataset/targeted_images_v3",
    "../../dataset/targeted_images_v4",
    "../../dataset/targeted_images_v5",
    "../../dataset/targeted_images_v6",
]

WIDTH = 265
HEIGHT = 67

ORANGE = (255,165,0)
BLACK  = (0,0,0)

CHARS = string.ascii_letters + string.digits + "@#="

CONFUSION_GROUPS = [
    "lI1",
    "O0",
    "B8",
    "S5",
    "Z2",
    "G6",
]

CONFUSION_MAP = {}
for g in CONFUSION_GROUPS:
    for c in g:
        CONFUSION_MAP[c] = list(g)

def random_text():
    return "".join(random.choice(CHARS) for _ in range(random.randint(4,6)))

def heavy_confusion_text():
    L = random.randint(4,6)
    t = [random.choice(CHARS) for _ in range(L)]

    k = random.randint(1,3)
    for _ in range(k):
        pos = random.randrange(L)
        grp = random.choice(CONFUSION_GROUPS)
        t[pos] = random.choice(grp)

    return "".join(t)

def swapped_variant_text():
    t = list(random_text())
    for i,c in enumerate(t):
        if c in CONFUSION_MAP and random.random() < 0.75:
            t[i] = random.choice(CONFUSION_MAP[c])
    return "".join(t)

def case_flip_text():
    t = random_text()
    return "".join(
        c.swapcase() if c.isalpha() and random.random()<0.6 else c
        for c in t
    )

def targeted_pattern_text():
    grp = random.choice(CONFUSION_GROUPS)
    core = "".join(random.choice(grp) for _ in range(random.randint(2,4)))
    pad = "".join(random.choice(CHARS) for _ in range(random.randint(2,3)))
    s = list(core + pad)
    random.shuffle(s)
    return "".join(s[:random.randint(4,6)])

def generate_targeted_text():
    r = random.random()

    if r < 0.30: return heavy_confusion_text()
    if r < 0.55: return swapped_variant_text()
    if r < 0.75: return targeted_pattern_text()
    if r < 0.90: return case_flip_text()
    return random_text()

def load_font():
    size = random.randint(46,54)
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

def jitter_pts(pts,j=10):
    return [(x+random.randint(-j,j), y+random.randint(-j,j)) for x,y in pts]

def dominant_trapezoid():
    y_top = random.randint(18,42)
    y_bot = random.randint(HEIGHT-8, HEIGHT+18)
    left  = random.randint(-70,25)
    right = random.randint(WIDTH-45, WIDTH+70)
    skew_top = random.randint(-30,25)
    skew_bot = random.randint(-25,35)

    pts=[
        (left,y_top),
        (right,y_top+skew_top),
        (right,y_bot+skew_bot),
        (left,y_bot)
    ]
    return jitter_pts(pts,12)

def draw_real_overlay(mask_draw):
    mask_draw.polygon(dominant_trapezoid(), fill=255)

    if random.random()<0.4:
        tri=[
            (-40,random.randint(0,HEIGHT)),
            (WIDTH+40,random.randint(-10,HEIGHT)),
            (WIDTH+40,random.randint(HEIGHT//2,HEIGHT+40))
        ]
        mask_draw.polygon(jitter_pts(tri,8), fill=255)

def draw_char_with_stroke(d,x,y,c,font):
    if random.random() < 0.5:
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            d.text((x+dx,y+dy),c,fill=255,font=font)
    d.text((x,y),c,fill=255,font=font)

def draw_text_mask(mask,text,font):

    d = ImageDraw.Draw(mask)

    boxes=[d.textbbox((0,0),c,font=font) for c in text]
    ws=[b[2]-b[0] for b in boxes]
    hs=[b[3]-b[1] for b in boxes]

    total_w=sum(ws)
    max_h=max(hs)

    x=(WIDTH-total_w)//2 + random.randint(-8,8)
    
    y0=(HEIGHT-max_h)//2 + random.randint(-6,10)

    for c,w in zip(text,ws):
        y=y0 + random.randint(-9,9)
        draw_char_with_stroke(d,x,y,c,font)
        x += w + random.randint(-4,4)

def edge_tint(img,text_mask):
    arr=np.array(img).astype(np.int16)
    m=np.array(text_mask)

    eroded = Image.fromarray(m).filter(ImageFilter.MinFilter(3))
    edge = m - np.array(eroded)

    ys,xs=np.where(edge>0)

    for y,x in zip(ys,xs):
        if random.random()<0.35:
            arr[y,x,0]+=random.randint(0,20)
            arr[y,x,1]+=random.randint(15,35)
            arr[y,x,2]-=random.randint(0,15)

    return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))

def generate_one(text,font,save_path):

    img = Image.new("RGB",(WIDTH,HEIGHT),ORANGE)

    overlay_mask = Image.new("L",(WIDTH,HEIGHT),0)
    draw_real_overlay(ImageDraw.Draw(overlay_mask))
    overlay_mask = overlay_mask.filter(ImageFilter.GaussianBlur(1.1))
    img.paste(BLACK,mask=overlay_mask)

    text_mask = Image.new("L",(WIDTH,HEIGHT),0)
    draw_text_mask(text_mask,text,font)
    text_mask = text_mask.filter(ImageFilter.GaussianBlur(0.5))

    orange_text_mask = ImageChops.multiply(text_mask,overlay_mask)
    black_text_mask  = ImageChops.subtract(text_mask,orange_text_mask)

    img.paste(BLACK,mask=black_text_mask)
    img.paste(ORANGE,mask=orange_text_mask)

    if random.random()<0.75:
        img=edge_tint(img,text_mask)

    if random.random()<0.6:
        img=img.filter(ImageFilter.GaussianBlur(0.35))

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
    used = load_existing_labels()
    print("Existing labels:",len(used))

    made=0

    while made < TOTAL_IMAGES:

        t = generate_targeted_text()
        if t in used:
            continue

        used.add(t)

        font = load_font()

        path=os.path.join(OUTPUT_DIR,f"{t}.png")
        generate_one(t,font,path)

        made+=1
        if made%100==0:
            print("Generated:",made)

    print("\nDone —",made,"new v5 samples")

if __name__=="__main__":
    main()
