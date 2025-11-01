from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
import cv2
import numpy as np
import os
import tempfile
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import shutil

app = Flask(__name__)

# ===========================
# Helper: Download single image
# ===========================
def download_image(url):
    try:
        print(f"Downloading image: {url}")
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}")
            return None
        image_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            print("❌ OpenCV decode failed")
        return img
    except Exception as e:
        print("Download exception:", e)
        return None


# ===========================
# Helper: Download all school images
# ===========================
def download_folder_images(folder_url, school_id):
    temp_dir = os.path.join("/tmp", f"school_{school_id}")
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        print(f"📁 Fetching folder: {folder_url}")
        html = requests.get(folder_url, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")

        count = 0
        for link in soup.find_all("a"):
            href = link.get("href")
            if href and href.lower().endswith((".jpg", ".jpeg", ".png")):
                file_url = urljoin(folder_url + "/", href)
                file_name = os.path.basename(href)
                try:
                    img_data = requests.get(file_url, timeout=20).content
                    with open(os.path.join(temp_dir, file_name), "wb") as f:
                        f.write(img_data)
                    count += 1
                except Exception as e:
                    print(f"⚠️ Failed {file_url}: {e}")
        print(f"✅ Downloaded {count} images to {temp_dir}")
        return temp_dir if count > 0 else None
    except Exception as e:
        print("Download folder error:", e)
        return None


# ===========================
# API: Find Similar Faces
# ===========================
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.json
        print("Incoming JSON:", data)

        # --- Validate ---
        if not data or "school_id" not in data or "folder_url" not in data or "image_url" not in data:
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        school_id = data["school_id"]
        folder_url = data["folder_url"]
        target_url = data["image_url"]

        # --- Download target ---
        target = download_image(target_url)
        if target is None:
            return jsonify({"error": "Failed to download target image"}), 400

        # --- Download folder images ---
        db_local = download_folder_images(folder_url, school_id)
        if not db_local:
            return jsonify({"error": "Failed to access folder URL"}), 400

        # --- DeepFace comparison ---
        print("🔍 Running DeepFace comparison...")
        try:
            results = DeepFace.find(
                img_path=target_url,
                db_path=db_local,
                enforce_detection=False,
                silent=True
            )
        except Exception as e:
            print("DeepFace error:", e)
            return jsonify({"error": f"DeepFace failed: {e}"}), 500

        if results is None or len(results) == 0:
            return jsonify({"similar_images": []})

        # --- Format output ---
        output = []
        for r in results[0].to_dict(orient="records"):
            output.append({
                "image_url": r.get("identity"),
                "similarity": round(1 - float(r.get("distance", 1.0)), 2)
            })

        return jsonify({"similar_images": output})

    except Exception as e:
        print("Unhandled exception:", e)
        return jsonify({"error": str(e)}), 500


# ===========================
# Root Route
# ===========================
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Talentify AI Face API is live 🚀"})


# ===========================
# Run App
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
