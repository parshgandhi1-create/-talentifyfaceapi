from flask import Flask, request, jsonify
from deepface import DeepFace
import requests, os, traceback, time, shutil
from io import BytesIO
from PIL import Image
from datetime import datetime

app = Flask(__name__)
TEMP_DIR = "temp_faces"
os.makedirs(TEMP_DIR, exist_ok=True)


# =========================================================
# ✅ Centralized image download handler with retries
# =========================================================
def download_image_safely(url, save_path, retries=3, timeout=20):
    """
    Downloads image from URL safely with retry and validation.
    Returns True if successful, False otherwise.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Talentify Face API)",
        "Accept": "image/*"
    }

    for attempt in range(1, retries + 1):
        try:
            print(f"⬇️ Attempt {attempt}: {url}")
            res = requests.get(url, headers=headers, timeout=timeout)

            if res.status_code != 200:
                print(f"⚠️ HTTP {res.status_code} - Retrying...")
                time.sleep(1)
                continue

            with open(save_path, "wb") as f:
                f.write(res.content)

            if os.path.getsize(save_path) < 1000:
                print("⚠️ File too small, skipping")
                os.remove(save_path)
                continue

            print(f"✅ Saved: {save_path}")
            return True

        except Exception as e:
            print(f"❌ Download error (attempt {attempt}): {e}")
            time.sleep(1)

    print(f"❌ Failed to download after {retries} attempts: {url}")
    return False


# =========================================================
# ✅ Auto-clean temp directory to prevent memory issues
# =========================================================
def manage_temp_folder(limit=25):
    try:
        if len(os.listdir(TEMP_DIR)) > limit:
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR)
            print("🧹 Temp folder cleaned up.")
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")


# =========================================================
# ✅ Basic headers for API responses
# =========================================================
@app.after_request
def add_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


# =========================================================
# ✅ Root route (health check)
# =========================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "version": "v3_optimized",
        "timestamp": datetime.utcnow().isoformat()
    })


# =========================================================
# ✅ Main face comparison route
# =========================================================
@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json(force=True)
        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not all([school_id, folder_url, image_url]):
            return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

        manage_temp_folder()

        # --------------------------------------------
        # Step 1️⃣  Download target image via proxy
        # --------------------------------------------
        proxy_target = f"https://talentify.co.in/school/image_proxy.php?url={image_url}"
        target_path = os.path.join(TEMP_DIR, "target_face.jpg")

        if not download_image_safely(proxy_target, target_path):
            return jsonify({"error": "Failed to download target image via proxy"}), 404

        target_img = Image.open(target_path)

        # --------------------------------------------
        # Step 2️⃣  Fetch list of school images
        # --------------------------------------------
        list_url = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        print(f"📂 Fetching list: {list_url}")

        folder_res = requests.get(list_url, timeout=15)
        if folder_res.status_code != 200:
            return jsonify({"error": f"Folder not accessible (HTTP {folder_res.status_code})"}), 404

        img_files = folder_res.json()
        if not img_files:
            return jsonify({"error": f"No images found for school {school_id}"}), 404

        print(f"🧠 Comparing {len(img_files)} images...")

        # --------------------------------------------
        # Step 3️⃣  Face comparison loop
        # --------------------------------------------
        best_match, best_score = None, 0.0

        for file_name in img_files:
            img_url = f"{folder_url}{file_name}"
            proxy_img_url = f"https://talentify.co.in/school/image_proxy.php?url={img_url}"
            temp_path = os.path.join(TEMP_DIR, f"compare_{file_name}")

            if not download_image_safely(proxy_img_url, temp_path):
                continue

            try:
                img = Image.open(temp_path)
                result = DeepFace.verify(target_img, img, enforce_detection=False)
                score = 1 - result["distance"]

                print(f"✅ Compared {file_name} | Score: {round(score, 3)}")

                if score > best_score:
                    best_score = score
                    best_match = img_url

            except Exception as e:
                print(f"❌ DeepFace error for {file_name}: {e}")
            finally:
                try:
                    os.remove(temp_path)
                except:
                    pass

        # --------------------------------------------
        # Step 4️⃣  Return best match
        # --------------------------------------------
        if not best_match:
            return jsonify({"error": "No match found"}), 404

        print(f"🏆 Best match: {best_match} | Score: {round(best_score, 3)}")

        return jsonify({
            "status": "success",
            "best_match": best_match,
            "score": round(float(best_score), 3)
        })

    except Exception as e:
        print("🔥 Traceback:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# =========================================================
# ✅ Run app
# =========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
