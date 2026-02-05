import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageChops

WIDTH = 265
HEIGHT = 67

ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)

CHARS = string.ascii_letters + string.digits + "@#="

# CONFIG

OUTPUT_DIR = "/Volumes/samsung_980/projects/captcha-reader/dataset/generated_samples"
TOTAL_IMAGES = 50000
FONT_PATH = "fonts/DejaVuSans.ttf"

# text

def random_text():
    length = random.randint(4, 6)
    return "".join(random.choice(CHARS) for _ in range(length))


def load_font(font_path=None, size=52):
    if font_path:
        return ImageFont.truetype(font_path, size)

    for f in [
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]:
        try:
            return ImageFont.truetype(f, size)
        except:
            continue

    return ImageFont.load_default()

# overlay shapes

def random_block_shape():
    top_y = random.randint(-10, HEIGHT // 2)
    bottom_y = random.randint(HEIGHT // 2, HEIGHT + 10)

    left = random.randint(-40, 40)
    right = random.randint(WIDTH - 40, WIDTH + 40)

    skew1 = random.randint(-30, 30)
    skew2 = random.randint(-30, 30)

    return [
        (left, top_y),
        (right, top_y + skew1),
        (right, bottom_y + skew2),
        (left, bottom_y)
    ]


def random_wedge():
    y = random.randint(0, HEIGHT)
    return [
        (-30, y),
        (WIDTH + 40, y + random.randint(-20, 20)),
        (WIDTH + 40, y + random.randint(25, 60)),
    ]


def random_poly():
    pts = []
    cx = random.randint(0, WIDTH)
    cy = random.randint(0, HEIGHT)

    for _ in range(random.randint(4, 6)):
        pts.append((
            cx + random.randint(-160, 160),
            cy + random.randint(-80, 80)
        ))
    return pts


def draw_overlay_mask(draw):
    for _ in range(random.randint(1, 2)):
        shape = random.choice([
            random_block_shape,
            random_wedge,
            random_poly
        ])
        draw.polygon(shape(), fill=255)

# generator

def generate_captcha(text, font_path, save_path):

    font = load_font(font_path, size=52)

    img = Image.new("RGB", (WIDTH, HEIGHT), ORANGE)

    # overlay first
    overlay_mask = Image.new("L", (WIDTH, HEIGHT), 0)
    draw_overlay_mask(ImageDraw.Draw(overlay_mask))
    img.paste(BLACK, mask=overlay_mask)

    # text mask
    text_mask = Image.new("L", (WIDTH, HEIGHT), 0)
    d = ImageDraw.Draw(text_mask)

    bbox = d.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    tx = (WIDTH - tw) // 2 + random.randint(-6, 6)
    ty = (HEIGHT - th) // 2 + random.randint(-3, 3)

    d.text((tx, ty), text, fill=255, font=font)

    orange_text_mask = ImageChops.multiply(text_mask, overlay_mask)
    black_text_mask = ImageChops.subtract(text_mask, orange_text_mask)

    img.paste(BLACK, mask=black_text_mask)
    img.paste(ORANGE, mask=orange_text_mask)

    img.save(save_path)

# batch generation

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    used = set()
    labels = []

    while len(used) < TOTAL_IMAGES:
        text = random_text()

        if text in used:
            continue

        used.add(text)

        filename = f"{text}.png"
        path = os.path.join(OUTPUT_DIR, filename)

        generate_captcha(text, FONT_PATH, path)
        labels.append((filename, text))

        # if len(used) % 50 == 0:
        #     print("Generated:", len(used))

    # optional label file (very useful)
    with open(os.path.join(OUTPUT_DIR, "labels.csv"), "w") as f:
        for name, text in labels:
            f.write(f"{name},{text}\n")

    print("\nDone — generated", len(used), "unique captchas")


# -------------------------

if __name__ == "__main__":
    main()
