from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
import cv2
import numpy as np
import os

app = Flask(__name__)

# ================
# Helper: Download Image
# ================
def download_image(url):
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            print(f"Download failed: {response.status_code}")
            return None
        image_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            print("OpenCV failed to decode image")
        return img
    except Exception as e:
        print("Download exception:", e)
        return None


# ================
# Route: Find Similar
# ================
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.json
        print("Incoming JSON:", data)

        # Validate parameters
        if not data or "school_id" not in data or "folder_url" not in data or "image_url" not in data:
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        folder_url = data["folder_url"]
        target_url = data["image_url"]

        # Download target image
        target = download_image(target_url)
        if target is None:
            return jsonify({"error": "Failed to download target image"}), 400

        # Collect reference images
        reference_images = []
        try:
            response = requests.get(folder_url)
            if response.status_code != 200:
                return jsonify({"error": "Failed to access folder URL"}), 400
        except Exception as e:
            print("Folder fetch error:", e)
            return jsonify({"error": "Failed to access folder URL"}), 400

        # Compare using DeepFace
        try:
            results = DeepFace.find(
                img_path=target_url,
                db_path=folder_url,
                enforce_detection=False,
                silent=True
            )
        except Exception as e:
            print("DeepFace error:", e)
            return jsonify({"error": f"DeepFace failed: {e}"}), 500

        if results is None or len(results) == 0:
            return jsonify({"similar_images": []})

        # Normalize results
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


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Talentify API is live"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
