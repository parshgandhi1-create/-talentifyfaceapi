from flask import Flask, request, jsonify
from deepface import DeepFace
import cv2, numpy as np, requests, os, re
from io import BytesIO
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "version": "url_final_render_proxy_headers"})

@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not all([school_id, folder_url, image_url]):
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        # ✅ Route image through proxy
        proxy_base = "https://talentify.co.in/school/image_proxy.php?url="
        proxied_url = f"{proxy_base}{image_url}"

        # ✅ Add headers to mimic browser (fixes 406)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "image/*,*/*;q=0.8"
        }

        # ✅ Download target image through proxy
        target_res = requests.get(proxied_url, headers=headers, timeout=10)
        if target_res.status_code != 200:
            return jsonify({"error": f"Failed to download target image via proxy ({target_res.status_code})"}), 404

        # ✅ Save target image temporarily
        target_path = f"temp_target_{school_id}.jpg"
        with open(target_path, "wb") as f:
            f.write(target_res.content)

        # ✅ Verify DeepFace can read this image
        try:
            target_repr = DeepFace.represent(img_path=target_path, model_name="Facenet", enforce_detection=True)
        except Exception as e:
            os.remove(target_path)
            return jsonify({"error": f"No face found in target image ({str(e)})"}), 400

        # ✅ Fetch folder contents
        folder_res = requests.get(folder_url, timeout=10)
        if folder_res.status_code != 200:
            os.remove(target_path)
            return jsonify({"error": f"Folder not accessible: {folder_url}"}), 404

        soup = BeautifulSoup(folder_res.text, "html.parser")
        img_urls = [
            folder_url + href
            for href in re.findall(r'photo_[^"\'<>]+\.(?:jpg|jpeg|png|webp)', folder_res.text, re.IGNORECASE)
        ]
        if not img_urls:
            os.remove(target_path)
            return jsonify({"error": f"No images found in {folder_url}"}), 404

        # ✅ Compare all images
        best_match = None
        best_score = 0.0

        for img_url in img_urls:
            try:
                proxied_img_url = f"{proxy_base}{img_url}"
                img_res = requests.get(proxied_img_url, headers=headers, timeout=10)
                if img_res.status_code != 200:
                    continue

                temp_img_path = f"temp_{os.path.basename(img_url)}"
                with open(temp_img_path, "wb") as f:
                    f.write(img_res.content)

                result = DeepFace.verify(
                    img1_path=target_path,
                    img2_path=temp_img_path,
                    model_name="Facenet",
                    enforce_detection=False
                )

                if result["verified"]:
                    similarity = 1 - result["distance"]
                    if similarity > best_score:
                        best_score = similarity
                        best_match = img_url

                os.remove(temp_img_path)
            except Exception:
                continue

        os.remove(target_path)

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
