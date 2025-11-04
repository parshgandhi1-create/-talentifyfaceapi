from flask import Flask, request, jsonify
import requests, cv2, numpy as np, os, re, gc, tempfile
from bs4 import BeautifulSoup

# ✅ Keep TensorFlow quiet & small
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "version": "url_final_render_optimized"})


@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not all([school_id, folder_url, image_url]):
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        # ✅ Proxy-based image download
        proxy_base = "https://talentify.co.in/school/image_proxy.php?url="
        target_proxy_url = proxy_base + image_url

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "image/*,*/*;q=0.8"
        }

        # ✅ Download target image
        target_res = requests.get(target_proxy_url, headers=headers, timeout=15)
        if target_res.status_code != 200:
            return jsonify({"error": f"Failed to download target image via proxy ({target_res.status_code})"}), 404

        # ✅ Load target image into memory (resized)
        target_arr = np.frombuffer(target_res.content, np.uint8)
        target_img = cv2.imdecode(target_arr, cv2.IMREAD_COLOR)
        target_img = cv2.resize(target_img, (400, 400))

        # ✅ Lazy import DeepFace to save RAM
        from deepface import DeepFace

        # ✅ Fetch folder HTML and extract image URLs
        folder_res = requests.get(folder_url, timeout=10)
        if folder_res.status_code != 200:
            return jsonify({"error": f"Folder not accessible: {folder_url}"}), 404

        soup = BeautifulSoup(folder_res.text, "html.parser")
        img_urls = [
            folder_url + href
            for href in re.findall(r'photo_[^"\'<>]+\.(?:jpg|jpeg|png|webp)', folder_res.text, re.IGNORECASE)
        ]
        if not img_urls:
            return jsonify({"error": f"No images found in {folder_url}"}), 404

        # ✅ Limit comparisons (to fit under 512 MB)
        img_urls = img_urls[:10]

        best_match = None
        best_score = 0.0

        for img_url in img_urls:
            try:
                proxied_img_url = proxy_base + img_url
                img_res = requests.get(proxied_img_url, headers=headers, timeout=10)
                if img_res.status_code != 200:
                    continue

                img_arr = np.frombuffer(img_res.content, np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                img = cv2.resize(img, (400, 400))

                # ✅ Save temporarily to disk for DeepFace
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_target, \
                     tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:

                    cv2.imwrite(tmp_target.name, target_img)
                    cv2.imwrite(tmp_img.name, img)

                    result = DeepFace.verify(
                        img1_path=tmp_target.name,
                        img2_path=tmp_img.name,
                        model_name="VGG-Face",
                        enforce_detection=False
                    )

                    score = 1 - result["distance"]
                    if score > best_score:
                        best_score = score
                        best_match = img_url

                # ✅ Cleanup temporary files + memory
                os.remove(tmp_target.name)
                os.remove(tmp_img.name)
                del img, img_arr
                gc.collect()

            except Exception:
                continue

        if not best_match:
            return jsonify({"error": "No match found"}), 404

        return jsonify({
            "status": "success",
            "best_match": best_match,
            "score": round(float(best_score), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
