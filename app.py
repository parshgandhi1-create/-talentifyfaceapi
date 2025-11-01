from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
import os
import tempfile

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status": "Talentify Face API is running ✅"})

@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json()
        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not folder_url or not image_url or not school_id:
            return jsonify({"error": "Missing parameters"}), 400

        # Download target image
        target_path = tempfile.mktemp(suffix=".jpg")
        with open(target_path, "wb") as f:
            f.write(requests.get(image_url).content)

        # Fetch list of all images in folder
        list_api = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        resp = requests.get(list_api)
        images = resp.json()

        if not isinstance(images, list):
            return jsonify({"error": "Invalid folder response", "details": images}), 500

        similar = []

        # Compare with each image
        for img_name in images:
            img_url = f"{folder_url}/{img_name}"
            try:
                temp_img = tempfile.mktemp(suffix=".jpg")
                with open(temp_img, "wb") as f:
                    f.write(requests.get(img_url).content)

                result = DeepFace.verify(
                    img1_path=target_path,
                    img2_path=temp_img,
                    model_name="VGG-Face",
                    enforce_detection=False
                )

                if result.get("verified"):
                    similar.append(img_name)

            except Exception as e:
                print("Error comparing", img_name, ":", str(e))
                continue

        return jsonify({"similar_images": similar})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
