@app.route('/find_similar', methods=['POST'])
def find_similar():
    data = request.get_json()

    # ✅ Validate parameters
    if not data or "school_id" not in data or "folder_url" not in data or "image_url" not in data:
        return jsonify({"error": "Missing parameters (school_id, folder_url, image_url)"}), 400

    school_id = data["school_id"]
    folder_url = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
    image_url = data["image_url"]

    print(f"[INFO] Request received for school_id={school_id}")
    print(f"[INFO] Target Image: {image_url}")
    print(f"[INFO] Folder URL: {folder_url}")

    # ===============================
    # 1️⃣ Download target image
    # ===============================
    try:
        resp = requests.get(image_url, timeout=15)
        resp.raise_for_status()
        target_img = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
        print("[Download] ✅ Target image downloaded successfully")
    except Exception as e:
        print(f"[Download] ❌ Failed: {e}")
        return jsonify({"error": "Failed to download target image"}), 400

    # ===============================
    # 2️⃣ Get candidate list from list_images.php
    # ===============================
    try:
        res = requests.get(folder_url, timeout=15)
        res.raise_for_status()
        files = res.json()
        if not isinstance(files, list) or len(files) == 0:
            raise Exception("No images found in folder")
        print(f"[Folder] ✅ Retrieved {len(files)} image names from list_images.php")
    except Exception as e:
        print(f"[Folder] ❌ Could not load image list: {e}")
        return jsonify({"error": "No candidate images found for comparison"}), 400

    # ===============================
    # 3️⃣ Download candidate images to temporary folder
    # ===============================
    base_path = f"/opt/render/.deepface/schools/{school_id}"
    os.makedirs(base_path, exist_ok=True)
    candidates = []

    for filename in files:
        img_url = f"https://talentify.co.in/uploads/schools/{school_id}/{filename}"
        save_path = os.path.join(base_path, filename)

        if not os.path.exists(save_path):
            try:
                r = requests.get(img_url, timeout=10)
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"[Skip] Failed to download {img_url}: {e}")
                continue
        candidates.append(save_path)

    if not candidates:
        print("[Folder] ⚠️ No local candidate images found after download")
        return jsonify({"error": "No candidate images found for comparison"}), 400

    print(f"[Folder] ✅ {len(candidates)} candidate images ready for comparison")

    # ===============================
    # 4️⃣ Compare embeddings
    # ===============================
    best_match = None
    best_score = float('inf')

    for img_path in candidates:
        try:
            result = DeepFace.verify(target_img, img_path, model_name="Facenet", enforce_detection=False)
            dist = result.get("distance", 1.0)
            print(f"[Compare] {os.path.basename(img_path)} → {dist:.4f}")
            if dist < best_score:
                best_score = dist
                best_match = os.path.basename(img_path)
        except Exception as e:
            print(f"[Compare Error] {img_path}: {e}")

    if not best_match:
        return jsonify({"match": None, "score": None, "status": "no_match"})

    print(f"[Result] ✅ Best match: {best_match} (score={best_score})")
    return jsonify({
        "school_id": school_id,
        "best_match": best_match,
        "score": best_score,
        "status": "success"
    })
