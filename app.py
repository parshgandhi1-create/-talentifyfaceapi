# ===================================
# Talentify Face API — Base64 Simplified (Final Render Version)
# ===================================
from flask import Flask, request, jsonify
from deepface import DeepFace
import os, base64, cv2, numpy as np

app = Flask(__name__)

# ---------- CONFIG ----------
UPLOAD_ROOT = "/tmp/schools"  # ✅ Safe for Render free plan
os.makedirs(UPLOAD_ROOT, exist_ok=True)


# ---------- HELPER: Decode Base64 Image ----------
def decode_base64_image(image_base64):
    try:
        image_bytes = base64.b64decode(image_base64.split(",")[-1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print("❌ Base64 decode failed:", e)
        return None


# ---------- API: FIND SIMILAR ----------
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json()
        school_id = data.get("school_id")
        image_base64 = data.get("image_base64") or data.get("target_base64")  # ✅ support both keys

        if not school_id or not image_base64:
            return jsonify({"error": "Missing parameters (school_id, image_base64)"}), 400

        # Decode target image
        target_img = decode_base64_image(image_base64)
        if target_img is None:
            return jsonify({"error": "Invalid base64 image"}), 400

        # Ensure school folder exists
        folder_path = os.path.join(UPLOAD_ROOT, str(school_id))
        os.makedirs(folder_path, exist_ok=True)

        # Collect all local images
        valid_ext = (".jpg", ".jpeg", ".png", ".webp")
        all_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(valid_ext)
        ]

        if not all_files:
            return jsonify({"error": f"No images found in school folder {folder_path}"}), 404

        print(f"📁 Comparing with {len(all_files)} images in {folder_path}")

        # Save temp image
        temp_path = "/tmp/input_face.jpg"
        cv2.imwrite(temp_path, target_img)

        best_match, best_score = None, 9999
        for file_path in all_files:
            try:
                result = DeepFace.verify(
                    img1_path=temp_path,
                    img2_path=file_path,
                    model_name="VGG-Face",
                    enforce_detection=False
                )
                distance = result.get("distance", 9999)
                if distance < best_score:
                    best_score = distance
                    best_match = os.path.basename(file_path)
            except Exception as e:
                print("⚠️ Skipped:", file_path, e)

        if best_match:
            return jsonify({
                "status": "success",
                "best_match": best_match,
                "score": round(best_score, 4)
            })
        else:
            return jsonify({"error": "No face match found"})

    except Exception as e:
        print("❌ Server error:", e)
        return jsonify({"error": str(e)}), 500


# ---------- HEALTH CHECK ----------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "version": "base64_final_render"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
