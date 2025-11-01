from flask import Flask, request, jsonify
from deepface import DeepFace
import tempfile
import requests
import random
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # ✅ allows PHP or frontend proxy access (CORS-safe)

# ==========================================
# ✅ Health Check
# ==========================================
@app.route("/")
def home():
    return jsonify({"status": "Talentify FaceMatch API (Indian Faces Model) ✅"})


# ==========================================
# 🔍 FIND SIMILAR FACES (ArcFace Optimized for Indian/Asian Faces)
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

        # ------------------------------------------
        # 🖼️ Download the target image safely
        # ------------------------------------------
        try:
            target_path = tempfile.mktemp(suffix=".jpg")
            headers = {"User-Agent": "Mozilla/5.0 (FaceMatchAgent)"}
            res = requests.get(image_url, headers=headers, timeout=15)
            res.raise_for_status()
            with open(target_path, "wb") as f:
                f.write(res.content)
        except Exception as e:
            return jsonify({"error": f"Failed to download target image: {e}"}), 500

        # ------------------------------------------
        # 📂 Fetch folder image list from your PHP endpoint
        # ------------------------------------------
        list_api = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        try:
            headers = {"User-Agent": "Mozilla/5.0 (FaceMatchAgent)"}
            r = requests.get(list_api, headers=headers, timeout=15)
            r.raise_for_status()
            images = r.json()
        except Exception as e:
            return jsonify({"error": f"Failed to fetch image list: {e}"}), 500

        if not isinstance(images, list):
            print("❌ Invalid response from list_images.php:", r.text[:300])
            return jsonify({"error": "Invalid response from PHP image list"}), 500

        # Limit to 40 images for speed
        if len(images) > 40:
            images = random.sample(images, 40)

        similar = []
        print(f"🧠 Comparing against {len(images)} photos using ArcFace...")

        # ------------------------------------------
        # 🤖 Face comparison loop (ArcFace + cosine)
        # ------------------------------------------
        for i, img_name in enumerate(images, 1):
            img_url = f"{folder_url}/{img_name}"
            try:
                tmp_path = tempfile.mktemp(suffix=".jpg")
                headers = {"User-Agent": "Mozilla/5.0 (FaceMatchAgent)"}
                r = requests.get(img_url, headers=headers, timeout=15)
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    f.write(r.content)

                result = DeepFace.verify(
                    img1_path=target_path,
                    img2_path=tmp_path,
                    model_name="ArcFace",       # ✅ best for Indian faces
                    distance_metric="cosine",   # ✅ stable for similar ethnicity
                    enforce_detection=False
                )

                distance = result.get("distance", 1.0)
                verified = result.get("verified", False)

                print(f"{i}/{len(images)} • {img_name} → {distance:.4f} • Match={verified}")

                # ✅ Threshold fine-tuned for Indian faces
                if distance < 0.65:
                    similar.append({
                        "image": img_name,
                        "distance": round(distance, 4)
                    })

            except Exception as e:
                print(f"⚠️ Error comparing {img_name}: {str(e)}")
                continue

        # ------------------------------------------
        # ✅ Return clean response to PHP proxy
        # ------------------------------------------
        if not similar:
            return jsonify({"message": "No similar faces found", "total_compared": len(images)})

        return jsonify({
            "similar_images": similar,
            "total_compared": len(images)
        })

    except Exception as e:
        print("💥 Exception:", str(e))
        return jsonify({"error": str(e)}), 500


# ==========================================
# 🚀 Render App Runner (Port 5000)
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
