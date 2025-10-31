from flask import Flask, request, jsonify
from deepface import DeepFace
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Talentify Face API is running!"

@app.route('/find_similar', methods=['POST'])
def find_similar():
    data = request.get_json()
    folder = data.get('folder')
    image_path = data.get('image')

    if not folder or not image_path:
        return jsonify({"error": "Missing folder or image"}), 400

    if not os.path.isdir(folder):
        return jsonify({"error": f"Folder not found: {folder}"}), 400

    target = os.path.join(folder, image_path)
    if not os.path.exists(target):
        return jsonify({"error": f"Target image not found: {target}"}), 400

    similar_images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            try:
                result = DeepFace.verify(target, img_path, model_name='Facenet', enforce_detection=False)
                if result['verified']:
                    similar_images.append(filename)
            except Exception as e:
                print("Error comparing:", filename, e)

    return jsonify({"similar_images": similar_images})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
