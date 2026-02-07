import base64
import sys

if len(sys.argv) != 2:
    print("Usage: python png_to_base64.py image.png")
    sys.exit(1)

path = sys.argv[1]

with open(path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

print(b64)
