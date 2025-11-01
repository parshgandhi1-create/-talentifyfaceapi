from flask import Flask, request, jsonify
import os, tempfile, requests, traceback
from deepface import DeepFace

app = Flask(__name__)

# =========================================
# 🧩 IMAGE DOWNLOAD FUNCTION
# =========================================
def download_image(image_url, save_path):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; TalentifyFaceAPI/1.0)",
            "Accept": "image/*,*/*;q=0.8"
        }
        response = requests.get(image_url, headers=headers, timeout=15, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '')
        if response.status_code == 200 and 'image' in content_type:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"⚠️ Invalid image response: {response.status_code}, {content_type}")
            return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


# =========================================
# 🔍 FIND SIMILAR FACES ENDPOINT
# =========================================
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not school_id or not folder_url or not image_url:
            return jsonify({"error": "Missing parameters"}), 400

        print(f"📩 Received request: school={school_id}, target={image_url}")

        # List all images from the school folder using the proxy
        list_url = f"{folder_url}/list_images.php?school_id={school_id}"
        image_list_response = requests.get(list_url, timeout=15)
        image_list = image_list_response.json().get("images", [])

        if not image_list:
            return jsonify({"error": "No images found in folder"}), 404

        with tempfile.TemporaryDirectory() as tmpdir:
            target_path = os.path.join(tmpdir, "target.jpg")

            # Download target image
            if not download_image(image_url, target_path):
                return jsonify({"error": f"Failed to download target image: {image_url}"}), 400

            results = []
            for candidate in image_list:
                candidate_url = f"{folder_url}/{candidate}"
                candidate_path = os.path.join(tmpdir, os.path.basename(candidate))

                if not download_image(candidate_url, candidate_path):
                    continue

                try:
                    # Compare faces (Facenet512 → accurate for Indian/Asian faces)
                    verify = DeepFace.verify(
                        img1_path=target_path,
                        img2_path=candidate_path,
                        model_name="Facenet512",
                        distance_metric="cosine",
                        enforce_detection=False
                    )
                    distance = verify.get("distance", 1.0)
                    verified = verify.get("verified", False)
                    if verified or distance < 0.35:
                        results.append({
                            "image_url": candidate_url,
                            "similarity": round(1 - distance, 3)
                        })
                except Exception as e:
                    print(f"⚠️ Error comparing {candidate_url}: {e}")
                    continue

        if not results:
            return jsonify({"message": "No similar faces found."})
        return jsonify({"similar_images": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "✅ Talentify Face API (Asian Model) is running."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
