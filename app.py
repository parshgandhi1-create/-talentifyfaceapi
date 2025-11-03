from flask import Flask, request, jsonify
import requests, numpy as np, os
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

    if not data or "school_id" not in data or "folder_url" not in data or "image_url" not in data:
        return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

    school_id = data["school_id"]
    folder_url = data["folder_url"].rstrip('/')
    image_url = data["image_url"]

    # 1️⃣ Download target image
    print(f"[Download] Trying: {image_url}")
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        target_img = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
    except Exception as e:
        print(f"[Download] ❌ Failed: {e}")
        return jsonify({"error": "Failed to download target image"}), 400

    print("[Download] ✅ Target image loaded successfully")

    # 2️⃣ Fetch candidate filenames from list_images.php
    list_url = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
    try:
        list_resp = requests.get(list_url, timeout=10)
        list_resp.raise_for_status()
        candidates_list = list_resp.json()
    except Exception as e:
        print(f"[List] ❌ Failed to fetch list: {e}")
        return jsonify({"error": "Failed to fetch image list"}), 400

    if not candidates_list:
        print("[List] ❌ No candidate images found")
        return jsonify({"error": "No candidate images found"}), 404

    print(f"[List] ✅ Found {len(candidates_list)} candidates")

    # 3️⃣ Compare embeddings
    best_match = None
    best_score = float('inf')

    for name in candidates_list:
        img_proxy_url = f"https://talentify.co.in/school/image_proxy.php?url=https://talentify.co.in/uploads/schools/{school_id}/{name}"

        try:
            resp = requests.get(img_proxy_url, timeout=10)
            resp.raise_for_status()
            cand_img = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))

            result = DeepFace.verify(target_img, cand_img, model_name="Facenet", enforce_detection=False)
            dist = result.get("distance", 1.0)
            print(f"Compared {name} → dist={dist}")

            if dist < best_score:
                best_score = dist
                best_match = name

        except Exception as e:
            print(f"[Compare] Error on {name}: {e}")

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
