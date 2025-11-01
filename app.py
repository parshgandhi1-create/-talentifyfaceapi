from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
import os
import tempfile
import random

# ✅ Create Flask app first
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "Talentify Face API is running ✅"})

@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        print("📩 Incoming data:", data)

        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not folder_url or not image_url or not school_id:
            return jsonify({"error": "Missing parameters"}), 400

        # --- Download target image ---
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            target_path = tempfile.mktemp(suffix=".jpg")
            with open(target_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            return jsonify({"error": f"Failed to download target image: {str(e)}"}), 500

        # --- Get list of images from folder API ---
        list_api = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        try:
            resp = requests.get(list_api, timeout=10)
            resp.raise_for_status()
            images = resp.json()
        except Exception as e:
            return jsonify({"error": f"Failed to fetch image list: {str(e)}"}), 500

        if not isinstance(images, list):
            print("❌ Invalid response from list_images.php:", resp.text[:300])
            return jsonify({"error": "Invalid folder response"}), 500

        # --- Limit to 15 random images for speed ---
        if len(images) > 15:
            images = random.sample(images, 15)

        similar = []
        print(f"🧠 Comparing against {len(images)} images...")

        for i, img_name in enumerate(images, 1):
            img_url = f"{folder_url}/{img_name}"
            try:
                img_resp = requests.get(img_url, timeout=10)
                img_resp.raise_for_status()
                temp_img = tempfile.mktemp(suffix=".jpg")
                with open(temp_img, "wb") as f:
                    f.write(img_resp.content)

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


if __name__ == "__main__":
    # ✅ Required for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
