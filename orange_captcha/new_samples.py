import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

OUTPUT_DIR = "/Volumes/samsung_980/projects/captcha-reader/dataset/generated_samples_v7"
TOTAL_IMAGES = 35000

EXISTING_DIRS = [
    "../dataset/generated_samples_v2",
    "../dataset/generated_samples_v3",
    "../dataset/generated_samples_v4",
    "../dataset/generated_samples_v5",
    "../dataset/generated_samples_v6",
    "../dataset/targeted_images",
    "../dataset/targeted_images_v2",
]

WIDTH = 265
HEIGHT = 67

ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)

CHARS = string.ascii_letters + string.digits + "@#="

def random_text():
    return "".join(random.choice(CHARS) for _ in range(random.randint(4, 6)))

def load_font(size=50):
    for f in [
        "fonts/DejaVuSans.ttf",
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
            arr[y,x,0] += random.randint(0,20)   # red
            arr[y,x,1] += random.randint(15,35)  # green
            arr[y,x,2] -= random.randint(0,15)   # blue

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

    text_mask = text_mask.filter(ImageFilter.GaussianBlur(0.5))

    orange_text_mask = ImageChops.multiply(text_mask, overlay_mask)
    black_text_mask  = ImageChops.subtract(text_mask, orange_text_mask)

    img.paste(BLACK, mask=black_text_mask)
    img.paste(ORANGE, mask=orange_text_mask)

    if random.random() < 0.75:
        img = edge_tint(img, text_mask)

    if random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(0.35))

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

        t = random_text()
        if t in used:
            continue

        used.add(t)

        path = os.path.join(OUTPUT_DIR,f"{t}.png")
        generate_one(t,font,path)

        made += 1
        if made % 100 == 0:
            print("Generated:",made)

    print("\nDone —",made,"new samples")

if __name__ == "__main__":
    main()
