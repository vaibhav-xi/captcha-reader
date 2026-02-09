import os
import cv2
import numpy as np

SRC_DIR = "../dataset/generated_samples_v3"
OUT_DIR = "output_images"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = 40

# =========================
# LETTER COLOR MASK (HSV)
# =========================
def letter_mask(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # yellow/orange letter band
    lower = np.array([15, 120, 120])
    upper = np.array([45, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((3,3),np.uint8),
        iterations=2
    )

    return mask


# =========================
# PROJECTION SPLIT
# =========================
def split_projection(mask):

    col = np.sum(mask > 0, axis=0).astype(np.float32)
    col = cv2.GaussianBlur(col, (1,25), 0).ravel()

    thresh = np.max(col) * 0.25
    gaps = col < thresh

    splits = []
    inside = False
    start = 0

    for i,g in enumerate(gaps):
        if not inside and not g:
            start = i
            inside = True
        elif inside and g:
            splits.append((start,i))
            inside = False

    if inside:
        splits.append((start,len(gaps)))

    splits = [s for s in splits if s[1]-s[0] > 8]
    return splits


# =========================
# TILE EXTRACT
# =========================
def extract_tile(gray, x1, x2):

    crop = gray[:, x1:x2]

    ys, xs = np.where(crop < 240)
    if len(xs) == 0:
        return None

    y1,y2 = ys.min(), ys.max()
    crop = crop[y1:y2+1,:]

    h,w = crop.shape
    scale = min(TARGET/h, TARGET/w)
    nh,nw = int(h*scale), int(w*scale)

    resized = cv2.resize(crop, (nw,nh))

    canvas = np.full((TARGET,TARGET), 255, np.uint8)
    ys = (TARGET-nh)//2
    xs = (TARGET-nw)//2
    canvas[ys:ys+nh, xs:xs+nw] = resized

    return canvas


# =========================
# MAIN
# =========================
X=[]
y=[]

files = [f for f in os.listdir(SRC_DIR) if f.endswith(".png")]
print("Images:", len(files))

for f in files:

    label = os.path.splitext(f)[0]
    img = cv2.imread(os.path.join(SRC_DIR,f))
    if img is None:
        continue

    mask = letter_mask(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite(f"{OUT_DIR}/{label}.png", gray) 

    # splits = split_projection(mask)

    # if not (4 <= len(splits) <= 6):
    #     print("Skip", f, "found", len(splits))
    #     continue

    # if len(splits) != len(label):
    #     print("Mismatch", f)
    #     continue

    # for i,(x1,x2) in enumerate(splits):

    #     tile = extract_tile(gray, x1, x2)
    #     if tile is None:
    #         continue

    #     cv2.imwrite(f"{OUT_DIR}/{label}_{i}.png", tile)

    #     X.append(tile[...,None]/255.0)
    #     y.append(label[i])


X=np.array(X,np.float32)
y=np.array(y)

np.save("letters_X.npy",X)
np.save("letters_y.npy",y)

print("\nDONE")
print("Tiles:", X.shape)
