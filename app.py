from flask import Flask, request, jsonify
import os, requests, cv2, numpy as np
from deepface import DeepFace
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def home():
    return "Talentify Face API is running"

@app.route('/find_similar', methods=['POST'])
def find_similar():
    data = request.get_json()

    # ✅ Final parameter validation
    if not data or "school_id" not in data or "folder_url" not in data or "image_url" not in data:
        return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

    school_id = data["school_id"]
    folder_url = data["folder_url"].rstrip('/')
    image_url = data["image_url"]

    # ===============================
    # 1️⃣ Download target image
    # ===============================
    print(f"[Download] Trying: {image_url}")
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        target_img = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
    except Exception as e:
        print(f"[Download] ❌ Failed: {e}")
        return jsonify({"error": "Failed to download target image"}), 400

    print("[Download] ✅ Image loaded successfully")

    # ===============================
    # 2️⃣ Scan all candidate photos
    # ===============================
    print(f"[Folder] Checking: {folder_url}")
    candidates = []

    # Assume your PHP proxy serves direct accessible images with filenames
    base_path = f"/opt/render/.deepface/schools/{school_id}"
    os.makedirs(base_path, exist_ok=True)

    # (Optional) Preload sample from the proxy folder if you already mirror them locally
    for name in os.listdir(base_path):
        if name.lower().endswith(('.jpg', '.jpeg', '.png')):
            candidates.append(os.path.join(base_path, name))

    if not candidates:
        print("[Folder] ❌ Folder not accessible or empty")
        return jsonify({"error": "Failed to access folder URL"}), 400

    print(f"[Folder] ✅ Found {len(candidates)} candidates")

    # ===============================
    # 3️⃣ Compare embeddings
    # ===============================
    best_match = None
    best_score = float('inf')

    for img_path in candidates:
        try:
            result = DeepFace.verify(target_img, img_path, model_name="Facenet", enforce_detection=False)
            dist = result.get("distance", 1.0)
            if dist < best_score:
                best_score = dist
                best_match = os.path.basename(img_path)
        except Exception as e:
            print(f"Error comparing {img_path}: {e}")

    if not best_match:
        return jsonify({"match": None, "score": None, "status": "no_match"})

    print(f"[Result] ✅ Best match: {best_match} (score={best_score})")
    return jsonify({
        "school_id": school_id,
        "best_match": best_match,
        "score": best_score,
        "status": "success"
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
