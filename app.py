from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2, numpy as np, requests, os, re
from bs4 import BeautifulSoup
from io import BytesIO

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "version": "url_final_render"})

@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not all([school_id, folder_url, image_url]):
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        # ✅ Add headers to prevent 404/406 on proxy
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        # ✅ Download target image
        target_res = requests.get(image_url, headers=headers, timeout=10)
        if target_res.status_code != 200:
            print("❌ Failed to fetch:", image_url, "Status:", target_res.status_code)
            return jsonify({
                "error": "Failed to download target image",
                "response": f"HTTP {target_res.status_code}"
            }), 404

        # Convert target image to numpy array (for DeepFace)
        target_arr = np.frombuffer(target_res.content, np.uint8)
        target_img = cv2.imdecode(target_arr, cv2.IMREAD_COLOR)

        # ✅ Fetch folder contents
        folder_res = requests.get(folder_url, headers=headers, timeout=10)
        if folder_res.status_code != 200:
            return jsonify({"error": f"Folder not accessible: {folder_url}"}), 404

        soup = BeautifulSoup(folder_res.text, "html.parser")
        img_urls = [
            folder_url + href
            for href in re.findall(r'photo_[^"\'<>]+\.(?:jpg|jpeg|png|webp)', folder_res.text, re.IGNORECASE)
        ]
        if not img_urls:
            return jsonify({"error": f"No images found in {folder_url}"}), 404

        # ✅ Compare all images using DeepFace
        best_match = None
        best_score = 0.0

        for img_url in img_urls:
            try:
                img_res = requests.get(img_url, headers=headers, timeout=10)
                if img_res.status_code != 200:
                    continue

                img_arr = np.frombuffer(img_res.content, np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

                # Compare using DeepFace
                result = DeepFace.verify(target_img, img, enforce_detection=False)
                score = 1 - result["distance"] if "distance" in result else 0

                if score > best_score:
                    best_score = score
                    best_match = img_url

            except Exception as e:
                print("⚠️ Error comparing:", img_url, e)
                continue

        if not best_match:
            return jsonify({"error": "No match found"}), 404

        return jsonify({
            "status": "success",
            "best_match": best_match,
            "score": round(float(best_score), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
