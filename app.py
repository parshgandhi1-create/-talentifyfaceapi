from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
import cv2
import numpy as np
import os
from urllib.parse import unquote

app = Flask(__name__)

# ================================
# Helper: Download image safely
# ================================
def download_image(url):
    try:
        url = unquote(url)  # Decode ?url=https%3A%2F%2F...
        print(f"[Download] Trying: {url}")

        headers = {"User-Agent": "Talentify-Face/1.0"}
        response = requests.get(url, headers=headers, timeout=25)

        if response.status_code != 200:
            print(f"[Download] ❌ Failed with {response.status_code}")
            return None

        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print("[Download] ❌ OpenCV could not decode image")
        else:
            print("[Download] ✅ Image loaded successfully")
        return img
    except Exception as e:
        print("[Download] Exception:", e)
        return None


# ================================
# Route: /find_similar
# ================================
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        print("Incoming JSON:", data)

        # Validate input
        required = ["school_id", "folder_url", "image_url"]
        if not all(k in data and data[k] for k in required):
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        folder_url = data["folder_url"]
        target_url = data["image_url"]

        # Check target image
        target = download_image(target_url)
        if target is None:
            return jsonify({"error": f"Failed to download target image: {target_url}"}), 400

        # Ensure folder accessible
        print(f"[Folder] Checking: {folder_url}")
        try:
            response = requests.get(folder_url, timeout=15)
            if response.status_code != 200:
                print("[Folder] ❌ Folder not accessible")
                return jsonify({"error": "Failed to access folder URL"}), 400
        except Exception as e:
            print("[Folder] Exception:", e)
            return jsonify({"error": "Failed to access folder URL"}), 400

        # Perform face comparison
        try:
            print("[DeepFace] Starting comparison...")
            results = DeepFace.find(
                img_path=target_url,
                db_path=folder_url,
                enforce_detection=False,
                silent=True
            )
            print("[DeepFace] ✅ Completed")
        except Exception as e:
            print("[DeepFace] ❌ Error:", e)
            return jsonify({"error": f"DeepFace failed: {str(e)}"}), 500

        if results is None or len(results) == 0:
            return jsonify({"similar_images": []})

        # Normalize results
        output = []
        df = results[0]
        for _, row in df.iterrows():
            identity = row.get("identity", "")
            distance = float(row.get("distance", 1.0))
            similarity = round(1 - distance, 2)
            output.append({
                "image_url": identity,
                "similarity": similarity
            })

        return jsonify({"similar_images": output})

    except Exception as e:
        print("[Unhandled Exception]", e)
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Talentify AI API live 🚀"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
