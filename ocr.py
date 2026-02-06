import cv2
import numpy as np
import pytesseract
import pandas as pd
import re
from pathlib import Path
from sklearn.cluster import KMeans

INPUT_DIR = "test_images"
DEBUG_ROOT = "debug_steps"
CHAR_WHITELIST = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@=#"


def save(debug_dir, name, img):
    cv2.imwrite(str(debug_dir / f"{name}.png"), img)

def valid_word(t):
    return bool(re.fullmatch(r"[A-Za-z0-9@=#]{4,8}", t))


def text_cluster_mask(img, debug_dir):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    save(debug_dir,"02_thresh", th)

    flood = th.copy()
    flood = cv2.dilate(flood, np.ones((2,2),np.uint8))
    h,w = th.shape
    mask = np.zeros((h+2,w+2), np.uint8)

    cv2.floodFill(flood, mask, (0,0), 0)
    cv2.floodFill(flood, mask, (w-1,0), 0)
    cv2.floodFill(flood, mask, (0,h-1), 0)
    cv2.floodFill(flood, mask, (w-1,h-1), 0)

    save(debug_dir,"03_flood_removed", flood)

    return flood

def segment_characters(mask, debug_dir):

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)

    boxes=[]

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if 6 < h < 90 and 3 < w < 90:
            boxes.append((x,y,w,h))

    boxes.sort(key=lambda b:b[0])

    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for (x,y,w,h) in boxes:
        cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),1)

    save(debug_dir,"04_boxes",vis)

    chars=[]
    for i,(x,y,w,h) in enumerate(boxes):
        crop = mask[y:y+h, x:x+w]
        crop = cv2.copyMakeBorder(crop,6,6,6,6,
                                  cv2.BORDER_CONSTANT,value=0)
        save(debug_dir,f"05_char_{i}",crop)
        chars.append(crop)

    return chars

def ocr_word(img):
    return pytesseract.image_to_string(
        img,
        config=f"--oem 1 --psm 8 -c tessedit_char_whitelist={CHAR_WHITELIST}"
    ).strip()

def ocr_char(img):
    return pytesseract.image_to_string(
        img,
        config=f"--oem 1 --psm 10 -c tessedit_char_whitelist={CHAR_WHITELIST}"
    ).strip()

def solve_one(path):

    img = cv2.imread(str(path))
    name = path.stem
    debug_dir = Path(DEBUG_ROOT)/name
    debug_dir.mkdir(parents=True, exist_ok=True)

    save(debug_dir,"01_original",img)

    mask = text_cluster_mask(img, debug_dir)

    mask_pad = cv2.copyMakeBorder(mask,12,12,12,12,
                                  cv2.BORDER_CONSTANT,value=0)
    save(debug_dir,"06_mask_pad",mask_pad)

    word = ocr_word(mask_pad)

    if valid_word(word):
        return word,"word_ocr"

    chars = segment_characters(mask, debug_dir)

    letters=[]
    for c in chars:
        t = ocr_char(c)
        if t:
            letters.append(t[0])

    return "".join(letters),"char_fallback"

def run():

    paths = list(Path(INPUT_DIR).glob("*.png"))
    results=[]

    for p in paths:
        text,mode = solve_one(p)
        print(p.name,"→",text,f"[{mode}]")
        results.append({"file":p.name,"prediction":text,"mode":mode})

    pd.DataFrame(results).to_csv("results.csv",index=False)


if __name__ == "__main__":
    run()
