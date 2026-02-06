import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
import pandas as pd

INPUT_DIR = "test_images"
DEBUG_DIR = "debug_tm"
FONT_PATH = "orange_captcha/fonts/DejaVuSans.ttf"

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@=#"
TEMPLATE_SIZE = 48

def build_templates():

    font = ImageFont.truetype(FONT_PATH, 42)

    templates = {}

    for ch in CHARS:
        img = Image.new("L", (TEMPLATE_SIZE, TEMPLATE_SIZE), 0)
        d = ImageDraw.Draw(img)

        bbox = d.textbbox((0,0), ch, font=font)
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]

        d.text(
            ((TEMPLATE_SIZE-w)//2, (TEMPLATE_SIZE-h)//2),
            ch,
            255,
            font=font
        )

        arr = np.array(img)
        _, arr = cv2.threshold(arr, 10, 255, cv2.THRESH_BINARY)

        templates[ch] = arr

    return templates

def segment(mask):

    col_sum = mask.sum(axis=0)

    col_sum = col_sum / col_sum.max()

    cols = col_sum > 0.15

    boxes = []
    in_char = False
    start = 0

    for x, v in enumerate(cols):
        if v and not in_char:
            in_char = True
            start = x
        elif not v and in_char:
            in_char = False
            end = x

            if end - start > 5:
                boxes.append((start, end))

    if in_char:
        boxes.append((start, len(cols)-1))

    return boxes

def preprocess(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,2
    )

    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          np.ones((2,2),np.uint8))

    return th

def match_char(crop, templates):

    crop = cv2.resize(crop, (TEMPLATE_SIZE, TEMPLATE_SIZE))

    best = None
    best_score = -1

    for ch, tmpl in templates.items():
        res = cv2.matchTemplate(crop, tmpl,
                                cv2.TM_CCOEFF_NORMED)
        score = res[0][0]
        if score > best_score:
            best_score = score
            best = ch

    return best, best_score

def solve_one(path, templates):

    img = cv2.imread(str(path))
    mask = preprocess(img)

    boxes = segment(mask)

    letters = []

    for (x0, x1) in boxes:
        crop = mask[:, x0:x1]

        ys = np.where(crop.sum(axis=1) > 0)[0]
        if len(ys) == 0:
            continue

        crop = crop[ys[0]:ys[-1]+1, :]

        crop = cv2.copyMakeBorder(
            crop,6,6,6,6,
            cv2.BORDER_CONSTANT,value=0
        )

        ch, score = match_char(crop, templates)
        print("match", ch, "score", round(score,3))
        letters.append(ch)

    return "".join(letters)

def run():

    templates = build_templates()

    rows = []

    for p in Path(INPUT_DIR).glob("*.png"):
        text = solve_one(p, templates)
        print(p.name, "→", text)
        rows.append({"file":p.name,"prediction":text})

    pd.DataFrame(rows).to_csv("results.csv",index=False)

if __name__ == "__main__":
    run()
