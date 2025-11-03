from flask import Flask, request, jsonify
import os, requests, cv2, numpy as np
from deepface import DeepFace
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Talentify Face API is running successfully!"

@app.route('/find_similar', methods=['POST'])
def find_similar():
    data = request.get_json()

    # ----------------------------------------
    # 1️⃣ Validate Parameters
    # ----------------------------------------
    if not data or "school_id" not in data or "folder_url" not in data or "image_url" not in data:
        return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

    school_id = data["school_id"]
    folder_url = data["folder_url"].rstrip('/')
    image_url = data["image_url"]

    print(f"[INFO] Request received for school_id={school_id}")
    print(f"[INFO] Target Image: {image_url}")
    print(f"[INFO] Folder URL: {folder_url}")

    # ----------------------------------------
    # 2️⃣ Download Target Image (via image_proxy)
    # ----------------------------------------
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*"
        }
        resp = requests.get(image_url, headers=headers, timeout=20)
        resp.raise_for_status()
        target_img = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
        print("[Download] ✅ Target image downloaded successfully")
    except Exception as e:
        print(f"[Download] ❌ Failed to download image: {e}")
        return jsonify({"error": f"Failed to download target image: {str(e)}"}), 400

    # ----------------------------------------
    # 3️⃣ Load Candidate Images from Local Mirror Folder
    # ----------------------------------------
    base_path = f"/opt/render/.deepface/schools/{school_id}"
    os.makedirs(base_path, exist_ok=True)

    candidates = [
        os.path.join(base_path, name)
        for name in os.listdir(base_path)
        if name.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not candidates:
        print("[Folder] ⚠️ No local candidate images found")
        return jsonify({"error": "No candidate images found for comparison"}), 400

    print(f"[Folder] ✅ Found {len(candidates)} candidate(s) locally")

    # ----------------------------------------
    # 4️⃣ Compare Using DeepFace
    # ----------------------------------------
    best_match = None
    best_score = float('inf')

    for img_path in candidates:
        try:
            result = DeepFace.verify(
                target_img, img_path,
                model_name="Facenet",
                enforce_detection=False
            )
            dist = result.get("distance", 1.0)
            print(f"[Compare] {os.path.basename(img_path)} → distance={dist:.4f}")

            if dist < best_score:
                best_score = dist
                best_match = os.path.basename(img_path)
        except Exception as e:
            print(f"[Error] Failed comparing {img_path}: {e}")

    if not best_match:
        print("[Result] ❌ No match found")
        return jsonify({
            "school_id": school_id,
            "status": "no_match",
            "match": None,
            "score": None
        })

    print(f"[Result] ✅ Best match: {best_match} (score={best_score:.4f})")

    # ----------------------------------------
    # 5️⃣ Return JSON Response
    # ----------------------------------------
    return jsonify({
        "school_id": school_id,
        "best_match": best_match,
        "score": round(best_score, 4),
        "status": "success"
    })

# ----------------------------------------
# 🚀 Start Flask App
# ----------------------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
