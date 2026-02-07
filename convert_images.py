import os
from PIL import Image, ImageOps

def normalize_folder_to_png(input_dir, output_dir="debug_out"):
    os.makedirs(output_dir, exist_ok=True)

    converted = []

    for name in os.listdir(input_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext not in [".jpg", ".jpeg"]:
            continue

        in_path = os.path.join(input_dir, name)

        try:
            img = Image.open(in_path)

            try:
                img = ImageOps.exif_transpose(img)
            except:
                pass

            img = img.convert("RGB")

            base = os.path.splitext(name)[0]
            out_path = os.path.join(output_dir, f"{base}.png")

            print(out_path)

            img.save(out_path, format="PNG", optimize=False, compress_level=0)
            converted.append(out_path)

        except Exception as e:
            print(f"Failed on {name}: {e}")

    return converted

input_dir = "dataset/test_images"
normalize_folder_to_png(input_dir, output_dir=input_dir)