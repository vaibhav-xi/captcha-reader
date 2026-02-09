import base64
import requests
import sys
import os
import json

if len(sys.argv) != 2:
    print("Usage: python send_captcha.py image.png")
    sys.exit(1)

path = sys.argv[1]

with open(path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

image_name = os.path.basename(path)
name_no_ext = os.path.splitext(image_name)[0]

payload = {
    "base64": b64,
    "fileName": image_name
}

with open(f"{name_no_ext}.txt", "w") as out:
    json.dump(payload, out, indent=2)

print("Saved payload to:", f"{name_no_ext}.txt")

print("base64 String:", b64)

r = requests.post(
    "http://127.0.0.1:8000/predict_base64",
    json=payload,
    timeout=10
)

print("Status:", r.status_code)
print("Response:", r.text)
