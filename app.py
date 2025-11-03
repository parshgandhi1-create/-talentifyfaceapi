from flask import Flask, request, jsonify
import os
import requests
import tempfile
import shutil
from deepface import DeepFace

app = Flask(__name__)  # ✅ must be declared before any routes

@app.route('/')
def home():
    return "Talentify Face API is running"

@app.route('/find_similar', methods=['POST'])
def find_similar():
    try:
        # === STEP 1: Receive parameters ===
        school_id = request.form.get('school_id')
        target_image_url = request.form.get('target_image')

        if not school_id or not target_image_url:
            return jsonify({"error": "Missing parameters (school_id, target_image)"}), 400

        print(f"[INFO] Request received for school_id={school_id}")
        print(f"[INFO] Target Image: {target_image_url}")

        # === STEP 2: List images via your PHP endpoint ===
        folder_url = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        print(f"[INFO] Folder URL: {folder_url}")

        response = requests.get(folder_url, timeout=20)
        if response.status_code != 200:
            return jsonify({"error": f"Failed to fetch image list: {response.status_code}"}), 400

        image_list = response.json()
        if not image_list or len(image_list) == 0:
            print("[Folder] ⚠️ No local candidate images found")
            return jsonify({"error": "No candidate images found for comparison"}), 400

        # === STEP 3: Create temp directory ===
        base_dir = tempfile.mkdtemp(prefix="deepface_")
        target_path = os.path.join(base_dir, "target.jpg")

        # === STEP 4: Download target image ===
        try:
            target_resp = requests.get(target_image_url, stream=True, timeout=20)
            target_resp.raise_for_status()
            with open(target_path, 'wb') as f:
                shutil.copyfileobj(target_resp.raw, f)
            print("[Download] ✅ Target image downloaded successfully")
        except Exception as e:
            print(f"[Download] ❌ Failed to download target image: {e}")
            shutil.rmtree(base_dir, ignore_errors=True)
            return jsonify({"error": f"Failed to download target image: {str(e)}"}), 400

        # === STEP 5: Download candidate images ===
        candidate_paths = []
        for img_name in image_list:
            img_url = f"https://talentify.co.in/uploads/schools/{school_id}/{img_name}"
            candidate_path = os.path.join(base_dir, img_name)
            try:
                img_resp = requests.get(img_url, stream=True, timeout=20)
                img_resp.raise_for_status()
                with open(candidate_path, 'wb') as f:
                    shutil.copyfileobj(img_resp.raw, f)
                candidate_paths.append(candidate_path)
            except Exception as e:
                print(f"[Download] ❌ Failed for {img_url}: {e}")
                continue

        if not candidate_paths:
            shutil.rmtree(base_dir, ignore_errors=True)
            return jsonify({"error": "No candidate images successfully downloaded"}), 400

        print(f"[INFO] {len(candidate_paths)} candidate images ready for comparison")

        # === STEP 6: Run DeepFace similarity check ===
        best_match = None
        best_score = 9999

        for path in candidate_paths:
            try:
                result = DeepFace.verify(target_path, path, model_name='VGG-Face', enforce_detection=False)
                distance = result.get("distance", 9999)
                if distance < best_score:
                    best_score = distance
                    best_match = os.path.basename(path)
            except Exception as e:
                print(f"[DeepFace] ⚠️ Error comparing {path}: {e}")
                continue

        shutil.rmtree(base_dir, ignore_errors=True)

        if not best_match:
            return jsonify({"error": "No face match found"}), 400

        return jsonify({
            "best_match": best_match,
            "similarity_score": round(1 - best_score, 3)
        })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({"error": f"Face API returned HTTP 400", "response": str(e)}), 400


# === Run server locally or on Render ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
