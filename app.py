from flask import Flask, request, jsonify
from deepface import DeepFace
import requests, os
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# ============================================================
# ✅ SAFE IMAGE DOWNLOADER (via proxy + optional save path)
# ============================================================
def download_image_safely(url, save_path=None):
    try:
        # Always use proxy for safety
        proxy_url = f"https://talentify.co.in/school/image_proxy.php?url={url}"
        headers = {
            "User-Agent": "Mozilla/5.0 (FaceMatch/1.0)",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
        }
        res = requests.get(proxy_url, timeout=15, headers=headers)

        if res.status_code != 200:
            print(f"⚠️ Proxy download failed ({res.status_code}): {url}")
            return None

        img = Image.open(BytesIO(res.content)).convert("RGB")
        img = img.resize((512, 512))  # ✅ Resize to reduce memory

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)

        return img
    except Exception as e:
        print(f"❌ download_image_safely error for {url}: {e}")
        return None


# ============================================================
# ✅ HEALTH CHECK
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "version": "optimized_limit3"})


# ============================================================
# ✅ MAIN FACE MATCH ROUTE
# ============================================================
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not all([school_id, folder_url, image_url]):
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        print(f"📡 Received request: school_id={school_id}")

        # ✅ Download target image safely
        target_img = download_image_safely(image_url)
        if target_img is None:
            return jsonify({"error": "Failed to download target image via proxy"}), 404

        # ✅ Fetch image list via list_images.php
        list_url = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        headers = {"User-Agent": "Mozilla/5.0 (FaceMatch/1.0)"}
        folder_res = requests.get(list_url, timeout=15, headers=headers)

        if folder_res.status_code != 200:
            return jsonify({"error": f"Folder not accessible (API {folder_res.status_code})"}), 404

        img_files = folder_res.json()
        if not img_files:
            return jsonify({"error": f"No images found for school {school_id}"}), 404

        # ✅ Limit to only 3 images
        img_files = img_files[:3]
        print(f"🖼️ Limiting to first 3 images: {img_files}")

        best_match = None
        best_score = 0.0

        # ✅ Compare target with limited images
        for file_name in img_files:
            img_url = f"{folder_url}{file_name}"
            candidate_img = download_image_safely(img_url)
            if candidate_img is None:
                continue

            try:
                result = DeepFace.verify(
                    target_img, candidate_img,
                    enforce_detection=False,
                    model_name="Facenet"  # ✅ Lightweight model
                )
                score = 1 - result["distance"]

                if score > best_score:
                    best_score = score
                    best_match = img_url

                print(f"✅ Compared {file_name}: score={round(score,3)}")

            except Exception as e:
                print(f"⚠️ DeepFace error on {file_name}: {e}")
                continue

        if not best_match:
            return jsonify({"error": "No match found"}), 404

        return jsonify({
            "status": "success",
            "best_match": best_match,
            "score": round(float(best_score), 3)
        })

    except Exception as e:
        print(f"🔥 ERROR: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# ✅ START FLASK APP
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
