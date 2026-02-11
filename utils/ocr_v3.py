import os
from paddleocr import PaddleOCR

DATASET = "dataset/orange-samples"

ocr = PaddleOCR(use_textline_orientation=True, lang='en')

correct = 0
total = 0

for f in os.listdir(DATASET):
    if not f.endswith(".png"):
        continue

    gt = os.path.splitext(f)[0]

    r = ocr.predict(os.path.join(DATASET, f))

    texts = r[0]["rec_texts"]
    pred = "".join(texts)

    if pred == gt:
        correct += 1

    total += 1
    print(gt, "→", pred)

print("\nAccuracy:", correct / total)
