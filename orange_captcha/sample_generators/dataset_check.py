import os
import sys
import string
from collections import Counter, defaultdict

ALLOWED_CHARS = set(string.ascii_letters + string.digits + "@=#")
MIN_LEN = 3
MAX_LEN = 8

def check_dirs():
    
    dirs = [
        "../dataset/generated_samples_v4",
        "../dataset/generated_samples_v5",
        "../dataset/orange-samples",
        "../dataset/hard_negatives_new",
    ]

    all_labels = {}
    bad_files = []
    bad_chars_files = []
    length_bad = []
    empty_label = []
    dup_labels = defaultdict(list)

    total_files = 0

    for d in dirs:
        if not os.path.isdir(d):
            print("Not a directory:", d)
            continue

        print(f"\nScanning: {d}")

        for f in os.listdir(d):
            if not f.lower().endswith(".png"):
                continue

            total_files += 1
            path = os.path.join(d, f)
            label = os.path.splitext(f)[0]

            if not label:
                empty_label.append(path)
                continue

            if not (MIN_LEN <= len(label) <= MAX_LEN):
                length_bad.append((path, label))

            bad_chars = [c for c in label if c not in ALLOWED_CHARS]
            if bad_chars:
                bad_chars_files.append((path, label, bad_chars))

            dup_labels[label].append(path)

            all_labels[path] = label

    print("\n================ DATASET REPORT ================")
    print("Total PNG files:", total_files)

    print("\n--- Invalid charset ---")
    print("Files:", len(bad_chars_files))
    for p,l,bc in bad_chars_files[:20]:
        print("BAD CHAR:", bc, "|", l, "|", p)

    print("\n--- Bad length ---")
    print("Files:", len(length_bad))
    for p,l in length_bad[:20]:
        print("BAD LEN:", len(l), "|", l, "|", p)

    print("\n--- Empty label ---")
    print("Files:", len(empty_label))
    for p in empty_label[:20]:
        print(p)

    dup_only = {k:v for k,v in dup_labels.items() if len(v) > 1}

    print("\n--- Duplicate labels across dirs ---")
    print("Unique duplicated labels:", len(dup_only))

    shown = 0
    for label, paths in dup_only.items():
        print("\nDUP:", label)
        for p in paths:
            print(" ", p)
        shown += 1
        if shown >= 10:
            break

    counter = Counter()
    for l in all_labels.values():
        counter.update(l)

    print("\n--- Character frequency ---")
    for c,n in counter.most_common():
        print(f"{c}: {n}")

    print("\n===============================================")
    
    if not bad_chars_files and not empty_label:
        print("\nCharset SAFE for CTC")
    else:
        print("\nFix charset issues BEFORE training")

    if length_bad:
        print("Length anomalies present")

    if dup_only:
        print("Duplicate labels present (may reduce diversity)")

if __name__ == "__main__":

    check_dirs()
