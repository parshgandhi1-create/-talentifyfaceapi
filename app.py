@app.route("/find_similar", methods=["POST"])
def find_similar():
    try:
        data = request.get_json()
        print("📩 Incoming data:", data)  # 👈 Log the JSON request

        school_id = data.get("school_id")
        folder_url = data.get("folder_url")
        image_url = data.get("image_url")

        if not folder_url or not image_url or not school_id:
            print("❌ Missing parameters")
            return jsonify({"error": "Missing parameters"}), 400

        print("🎯 Downloading target image:", image_url)
        target_path = tempfile.mktemp(suffix=".jpg")
        with open(target_path, "wb") as f:
            f.write(requests.get(image_url).content)

        list_api = f"https://talentify.co.in/school/list_images.php?school_id={school_id}"
        resp = requests.get(list_api)
        print("📦 List API response:", resp.text[:300])  # Log first few chars

        images = resp.json()
        if not isinstance(images, list):
            print("❌ Invalid list_images response")
            return jsonify({"error": "Invalid folder response", "details": images}), 500

        similar = []

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
                print("🧠 Compared:", img_name, "->", result.get("verified"))

                if result.get("verified"):
                    similar.append(img_name)
            except Exception as e:
                print("⚠️ Error comparing", img_name, ":", e)
                continue

        print("✅ Found similar:", similar)
        return jsonify({"similar_images": similar})

    except Exception as e:
        print("💥 Exception:", str(e))
        return jsonify({"error": str(e)}), 500
