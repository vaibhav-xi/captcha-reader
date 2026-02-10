import base64
import requests
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python send_captcha.py image.png")
    sys.exit(1)

path = sys.argv[1]

with open(path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "base64": b64,
    "fileName": os.path.basename(path)
}

print("base64 String: ", b64)

r = requests.post(
    "http://127.0.0.1:8080/predict_base64",
    json=payload,
    timeout=10
)

print("Status:", r.status_code)
print("Response:", r.text)
