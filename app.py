from flask import Flask, request, jsonify
from deepface import DeepFace
import os

app = Flask(__name__)

@app.route('/find_similar', methods=['POST'])
def find_similar():
    data = request.get_json()
    folder = data.get("folder")
    image = data.get("image")

    if not folder or not image:
        return jsonify({"error": "Missing folder or image"}), 400

    if not os.path.isdir(folder) or not os.path.isfile(image):
        return jsonify({"error": "Invalid folder or image path"}), 400

    try:
        result = DeepFace.find(img_path=image, db_path=folder)
        similar_images = result[0]['identity'].tolist()
        return jsonify({"similar_images": similar_images})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Talentify Face Recognition API is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
