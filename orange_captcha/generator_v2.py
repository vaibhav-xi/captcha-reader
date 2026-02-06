import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageFilter

# CONFIG

OUTPUT_DIR = "/Volumes/samsung_980/projects/captcha-reader/dataset/generated_samples_v3"
TOTAL_IMAGES = 50000

WIDTH = 265
HEIGHT = 67

ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)

CHARS = string.ascii_letters + string.digits + "@#="

# text

def random_text():
    return "".join(random.choice(CHARS) for _ in range(random.randint(4, 6)))

# font

def load_font(size=50):
    for f in [
        "fonts/DejaVuSans.ttf",
        "/home/jovyan/projects/captcha-reader/orange_captcha/fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(f, size)
        except:
            continue
    return ImageFont.load_default()

# overlay shapes

def dominant_trapezoid():
    y_top = random.randint(20, 40)
    y_bot = random.randint(HEIGHT - 10, HEIGHT + 15)

    left = random.randint(-60, 20)
    right = random.randint(WIDTH - 40, WIDTH + 60)

    skew_top = random.randint(-25, 15)
    skew_bot = random.randint(-10, 30)

    return [
        (left, y_top),
        (right, y_top + skew_top),
        (right, y_bot + skew_bot),
        (left, y_bot)
    ]


def corner_wedge():
    y = random.randint(0, HEIGHT)
    return [
        (-40, y),
        (WIDTH + 40, y + random.randint(-15, 10)),
        (WIDTH + 40, y + random.randint(25, 55))
    ]


def draw_real_overlay(mask_draw):
    mask_draw.polygon(dominant_trapezoid(), fill=255)

    if random.random() < 0.35:
        mask_draw.polygon(corner_wedge(), fill=255)


# draw jittered text mask

def draw_text_mask(mask, text, font):

    d = ImageDraw.Draw(mask)

    glyph_boxes = [d.textbbox((0, 0), c, font=font) for c in text]
    glyph_widths = [b[2] - b[0] for b in glyph_boxes]
    glyph_heights = [b[3] - b[1] for b in glyph_boxes]

    total_w = sum(glyph_widths)
    max_h = max(glyph_heights)

    start_x = (WIDTH - total_w) // 2 + random.randint(-5, 5)
    base_y = (HEIGHT - max_h) // 2 + random.randint(-2, 2)

    x = start_x

    for c, gw in zip(text, glyph_widths):

        if random.random() < 0.55:
            y_offset = random.randint(-8, 8)
        else:
            y_offset = 0

        d.text((x, base_y + y_offset), c, fill=255, font=font)
        x += gw + random.randint(-2, 2)

# single sample generator

def generate_one(text, font, save_path):

    img = Image.new("RGB", (WIDTH, HEIGHT), ORANGE)

    overlay_mask = Image.new("L", (WIDTH, HEIGHT), 0)
    draw_real_overlay(ImageDraw.Draw(overlay_mask))
    img.paste(BLACK, mask=overlay_mask)

    text_mask = Image.new("L", (WIDTH, HEIGHT), 0)
    draw_text_mask(text_mask, text, font)

    orange_text_mask = ImageChops.multiply(text_mask, overlay_mask)
    black_text_mask = ImageChops.subtract(text_mask, orange_text_mask)

    img.paste(BLACK, mask=black_text_mask)
    img.paste(ORANGE, mask=orange_text_mask)

    if random.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(0.6))

    img.save(save_path, quality=random.randint(88, 95))

# batch driver

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font = load_font(50)

    used = set()

    while len(used) < TOTAL_IMAGES:

        text = random_text()
        if text in used:
            continue

        used.add(text)

        filename = f"{text}.png"
        path = os.path.join(OUTPUT_DIR, filename)

        generate_one(text, font, path)

        # if len(used) % 500 == 0:
        #     print("Generated:", len(used))

    print("\nDone — generated", len(used), "unique captchas")


# -------------------------

if __name__ == "__main__":
    main()
