from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
import os
import tempfile
import random

app = Flask(__name__)

# ==================================================
# ✅ Root Endpoint — Health Check
# ==================================================
@app.route("/")
def home():
    return jsonify({"status": "Talentify Face Match API is running ✅"})


# ==================================================
# 🔍 Find Similar Faces Endpoint
# ==================================================
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        print("📩 Incoming data:", data)

        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not all([school_id, folder_url, image_url]):
            return jsonify({"error": "Missing parameters"}), 400

        # ==================================================
        # 🖼️ Step 1 — Download Target Image via Proxy
        # ==================================================
        try:
            target_path = tempfile.mktemp(suffix=".jpg")
            r = requests.get(image_url, timeout=20)
            r.raise_for_status()
            with open(target_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            return jsonify({"error": f"Failed to download target image: {e}"}), 500

        # ==================================================
        # 📁 Step 2 — Get Image List via PHP API
        # ==================================================
        list_api = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        try:
            resp = requests.get(list_api, timeout=20)
            resp.raise_for_status()
            images = resp.json()
        except Exception as e:
            return jsonify({"error": f"Failed to fetch image list: {e}"}), 500

        if not isinstance(images, list):
            print("❌ Invalid response from list_images.php:", resp.text[:300])
            return jsonify({"error": "Invalid folder response"}), 500

        # ==================================================
        # ⚙️ Step 3 — Limit Sample for Faster Comparison
        # ==================================================
        if len(images) > 20:
            images = random.sample(images, 20)

        print(f"🧠 Comparing against {len(images)} images...")
        similar = []

        # ==================================================
        # 🧩 Step 4 — Compare Faces Using ArcFace + RetinaFace
        # ==================================================
        for i, img_name in enumerate(images, 1):
            img_url = f"{folder_url}/{img_name}"

            try:
                temp_img = tempfile.mktemp(suffix=".jpg")
                r = requests.get(img_url, timeout=15)
                r.raise_for_status()
                with open(temp_img, "wb") as f:
                    f.write(r.content)

                result = DeepFace.verify(
                    img1_path=target_path,
                    img2_path=temp_img,
                    model_name="ArcFace",          # Better for Indian/Asian faces
                    detector_backend="retinaface", # More accurate facial alignment
                    distance_metric="cosine",
                    enforce_detection=False
                )

                verified = result.get("verified")
                distance = result.get("distance", 1.0)
                print(f"{i}/{len(images)} 🔍 {img_name} -> verified={verified}, distance={distance:.3f}")

                if verified and distance < 0.45:  # Lower threshold = stricter match
                    similar.append({
                        "image": img_name,
                        "distance": distance
                    })

            except Exception as e:
                print("⚠️ Error comparing", img_name, ":", str(e))
                continue

        # ==================================================
        # 📦 Step 5 — Return Result
        # ==================================================
        if not similar:
            return jsonify({"message": "No valid response or no similar images found."}), 200

        print("✅ Done! Found similar:", similar)
        return jsonify({"similar_images": similar})

    except Exception as e:
        print("💥 Exception:", str(e))
        return jsonify({"error": str(e)}), 500


# ==================================================
# 🚀 Run (Render uses port 5000)
# ==================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
