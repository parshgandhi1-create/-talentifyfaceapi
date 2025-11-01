from flask import Flask, request, jsonify
import os, tempfile, requests, traceback
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)

# =========================================
# 🧩 SAFE IMAGE DOWNLOAD (handles 406 error)
# =========================================
def download_image(image_url, save_path):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; TalentifyFaceAPI/1.0)",
            "Accept": "image/*,*/*;q=0.8"
        }
        response = requests.get(image_url, headers=headers, timeout=15, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '')
        if response.status_code == 200 and 'image' in content_type:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"⚠️ Not an image: {response.status_code}, {content_type}")
            return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

# =========================================
# 🔍 FIND SIMILAR FACES
# =========================================
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        target_url = data.get("target_url")
        image_urls = data.get("image_urls", [])
        threshold = float(data.get("threshold", 0.35))

        if not target_url or not image_urls:
            return jsonify({"error": "Missing parameters"}), 400

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = os.path.join(tmpdir, "target.jpg")

            if not download_image(target_url, target_path):
                return jsonify({"error": f"Failed to download target image: {target_url}"}), 400

            results = []
            for url in image_urls:
                candidate_path = os.path.join(tmpdir, os.path.basename(url))
                if not download_image(url, candidate_path):
                    continue

                try:
                    # Compare using DeepFace tuned for Asian faces
                    verify = DeepFace.verify(
                        img1_path=target_path,
                        img2_path=candidate_path,
                        model_name="Facenet512",  # More accurate for Indian/Asian faces
                        distance_metric="cosine",
                        enforce_detection=False
                    )
                    distance = verify.get("distance", 1.0)
                    verified = verify.get("verified", False)

                    if verified or distance < threshold:
                        results.append({
                            "image_url": url,
                            "similarity_score": round(1 - distance, 3)
                        })
                except Exception as e:
                    print(f"⚠️ Comparison error: {e}")
                    continue

        if not results:
            return jsonify({"message": "No similar images found."})
        return jsonify({"matches": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "✅ Talentify Face API is running."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
