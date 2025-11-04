from flask import Flask, request, jsonify
from deepface import DeepFace
import requests, os, tempfile, signal, shutil
from io import BytesIO

app = Flask(__name__)

# ✅ Preload lightweight SFace model (fast, CPU friendly)
print("🔄 Preloading SFace model...")
MODEL = DeepFace.build_model("SFace")
print("✅ SFace model ready.")

# =============================
# Utility: safe image downloader
# =============================
def download_image_safely(url, save_path):
    """
    Download image with retries and proxy fallback.
    """
    try:
        # Direct download attempt
        res = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200 and res.content:
            with open(save_path, "wb") as f:
                f.write(res.content)
            return True

        # Fallback to image_proxy if direct failed
        proxy_url = f"https://talentify.co.in/school/image_proxy.php?url={url}"
        res = requests.get(proxy_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200 and res.content:
            with open(save_path, "wb") as f:
                f.write(res.content)
            return True
        else:
            print(f"⚠️ Proxy fetch failed ({res.status_code}) for {url}")
            return False

    except Exception as e:
        print(f"❌ Download failed for {url}: {e}")
        return False


# =============================
# Utility: timeout guard
# =============================
def timeout_handler(signum, frame):
    raise TimeoutError("⏰ DeepFace operation timed out")

signal.signal(signal.SIGALRM, timeout_handler)

# =============================
# Root route
# =============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "version": "render_final_v3"})


# =============================
# Main API route
# =============================
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not all([school_id, folder_url, image_url]):
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        print(f"📡 Received request: school_id={school_id}")

        # =============================
        # Step 1️⃣: Create temp folder
        # =============================
        temp_folder = tempfile.mkdtemp(prefix=f"school_{school_id}_")
        target_path = os.path.join(temp_folder, "target.jpg")

        # =============================
        # Step 2️⃣: Download target image
        # =============================
        if not download_image_safely(image_url, target_path):
            shutil.rmtree(temp_folder, ignore_errors=True)
            return jsonify({"error": "Failed to download target image via proxy"}), 404

        # =============================
        # Step 3️⃣: Get list of school images
        # =============================
        list_url = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        res = requests.get(list_url, timeout=10)
        if res.status_code != 200:
            shutil.rmtree(temp_folder, ignore_errors=True)
            return jsonify({"error": "Folder not accessible (API 406)"}), 404

        images = res.json()
        if not images:
            shutil.rmtree(temp_folder, ignore_errors=True)
            return jsonify({"error": "No images found"}), 404

        # Limit to 3 images for memory safety
        images = images[:3]
        print(f"🖼️ Limiting to first 3 images: {images}")

        # =============================
        # Step 4️⃣: Download each image locally
        # =============================
        local_images = []
        for img_name in images:
            img_url = f"{folder_url}{img_name}"
            save_path = os.path.join(temp_folder, img_name)
            if download_image_safely(img_url, save_path):
                local_images.append(save_path)

        if not local_images:
            shutil.rmtree(temp_folder, ignore_errors=True)
            return jsonify({"error": "No downloadable images found"}), 404

        # =============================
        # Step 5️⃣: Compare using DeepFace (SFace)
        # =============================
        best_match = None
        best_score = 0.0

        signal.alarm(25)  # ⏰ Limit operation to 25 seconds

        for img_path in local_images:
            try:
                result = DeepFace.verify(
                    img1_path=target_path,
                    img2_path=img_path,
                    model_name="SFace",
                    enforce_detection=False
                )
                score = float(result.get("similarity", 0.0))
                if score > best_score:
                    best_score = score
                    best_match = img_path
            except Exception as e:
                print(f"⚠️ Comparison failed for {img_path}: {e}")
                continue

        signal.alarm(0)

        # =============================
        # Step 6️⃣: Cleanup & Response
        # =============================
        shutil.rmtree(temp_folder, ignore_errors=True)

        if not best_match:
            return jsonify({"error": "No match found"}), 404

        return jsonify({
            "status": "success",
            "best_match": best_match.split("/")[-1],
            "score": round(best_score, 3)
        })

    except TimeoutError as te:
        print("⏰ Timeout: ", te)
        return jsonify({"error": "DeepFace processing timed out"}), 504

    except Exception as e:
        print("❌ Exception: ", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
