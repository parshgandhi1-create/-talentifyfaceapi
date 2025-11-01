from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
import os
import tempfile
import random

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "Talentify Face API is running ✅"})

@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json()
        print("📩 Incoming data:", data)

        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not folder_url or not image_url or not school_id:
            return jsonify({"error": "Missing parameters"}), 400

        # --- Download target image ---
        target_path = tempfile.mktemp(suffix=".jpg")
        with open(target_path, "wb") as f:
            f.write(requests.get(image_url, timeout=10).content)

        # --- Get list of images from folder API ---
        list_api = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        resp = requests.get(list_api, timeout=10)
        images = resp.json()

        if not isinstance(images, list):
            print("❌ Invalid response from list_images.php:", resp.text[:300])
            return jsonify({"error": "Invalid folder response"}), 500

        # --- Limit to 15 random images for speed (configurable) ---
        if len(images) > 15:
            images = random.sample(images, 15)

        similar = []
        print(f"🧠 Comparing against {len(images)} images...")

        for i, img_name in enumerate(images, 1):
            img_url = f"{folder_url}/{img_name}"
            try:
                temp_img = tempfile.mktemp(suffix=".jpg")
                with open(temp_img, "wb") as f:
                    f.write(requests.get(img_url, timeout=10).content)

                result = DeepFace.verify(
                    img1_path=target_path,
                    img2_path=temp_img,
                    model_name="Facenet",  # ⚡ Faster model
                    enforce_detection=False
                )

                print(f"{i}/{len(images)} 🔍 {img_name} -> {result.get('verified')}")
                if result.get("verified"):
                    similar.append(img_name)

            except Exception as e:
                print("⚠️ Error comparing", img_name, ":", str(e))
                continue

        print("✅ Done! Found similar:", similar)
        return jsonify({"similar_images": similar})

    except Exception as e:
        print("💥 Exception:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render looks for this port
    app.run(host="0.0.0.0", port=5000)
