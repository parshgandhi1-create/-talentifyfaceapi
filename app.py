from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
import tempfile
import random

app = Flask(__name__)

# ==========================================
# ✅ HEALTH CHECK
# ==========================================
@app.route("/")
def home():
    return jsonify({"status": "Talentify Face API is running ✅"})


# ==========================================
# 🔍 FIND SIMILAR FACES
# ==========================================
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        print("📩 Incoming data:", data)

        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not folder_url or not image_url or not school_id:
            return jsonify({"error": "Missing parameters"}), 400

        # ----------------------------
        # 🧱 Helper: Safe image downloader
        # ----------------------------
        def download_image(url):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Accept": "image/*,*/*;q=0.8",
                }
                r = requests.get(url, headers=headers, timeout=15)
                r.raise_for_status()
                temp_path = tempfile.mktemp(suffix=".jpg")
                with open(temp_path, "wb") as f:
                    f.write(r.content)
                return temp_path
            except Exception as e:
                raise RuntimeError(f"Failed to download image: {e}")

        # --- Download target image ---
        target_path = download_image(image_url)

        # ----------------------------
        # 🧱 Fetch image list from PHP
        # ----------------------------
        list_api = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json,text/*;q=0.8",
            }
            resp = requests.get(list_api, headers=headers, timeout=15)
            resp.raise_for_status()
            images = resp.json()
        except Exception as e:
            return jsonify({"error": f"Failed to fetch image list: {e}"}), 500

        if not isinstance(images, list):
            print("❌ Invalid response from list_images.php:", resp.text[:300])
            return jsonify({"error": "Invalid folder response"}), 500

        # --- Limit comparisons for performance ---
        if len(images) > 15:
            images = random.sample(images, 15)

        similar = []
        print(f"🧠 Comparing against {len(images)} images...")

        # ----------------------------
        # 🔍 Compare each image
        # ----------------------------
        for i, img_name in enumerate(images, 1):
            img_url = f"{folder_url}/{img_name}"
            try:
                temp_img = download_image(img_url)

                result = DeepFace.verify(
                    img1_path=target_path,
                    img2_path=temp_img,
                    model_name="Facenet",
                    enforce_detection=False
                )

                print(f"{i}/{len(images)} 🔍 {img_name} -> {result.get('verified')}")
                if result.get("verified"):
                    similar.append(img_name)

            except Exception as e:
                print(f"⚠️ Error comparing {img_name}: {e}")
                continue

        print("✅ Done! Found similar:", similar)
        return jsonify({"similar_images": similar})

    except Exception as e:
        print("💥 Exception:", str(e))
        return jsonify({"error": str(e)}), 500


# ==========================================
# 🚀 FLASK STARTUP (Render)
# ==========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
