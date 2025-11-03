from flask import Flask, request, jsonify
import os, requests, numpy as np
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

    # ✅ Parameter validation
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
    # 2️⃣ Fetch image list from list_images.php
    # ===============================
    list_api = f"{folder_url}/list_images.php?school_id={school_id}"
    try:
        list_resp = requests.get(list_api, timeout=10)
        list_resp.raise_for_status()
        candidates = list_resp.json()
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Empty or invalid JSON list")
    except Exception as e:
        print(f"[Folder] ❌ Failed to fetch list: {e}")
        return jsonify({"error": "Failed to get image list"}), 400

    print(f"[Folder] ✅ Found {len(candidates)} candidates")

    # ===============================
    # 3️⃣ Compare embeddings remotely
    # ===============================
    best_match = None
    best_score = float('inf')

    for filename in candidates:
        try:
            proxy_url = f"{folder_url}/image_proxy.php?url={folder_url.replace('/school', '/uploads/schools')}/{school_id}/{filename}"
            resp = requests.get(proxy_url, timeout=10)
            resp.raise_for_status()
            cand_img = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))

            result = DeepFace.verify(target_img, cand_img, model_name="Facenet", enforce_detection=False)
            dist = result.get("distance", 1.0)
            print(f"[Compare] {filename} → distance={dist:.4f}")

            if dist < best_score:
                best_score = dist
                best_match = filename
        except Exception as e:
            print(f"Error comparing {filename}: {e}")

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
