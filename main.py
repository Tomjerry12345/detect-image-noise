from flask import Flask, request
from flask_cors import CORS
import tensorflow as tf
import os
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

root_path = os.path.dirname(__file__)
model_path = os.path.join(root_path, "model.h5")

img_path = os.path.join(root_path, "uploads/image.jpg")

image_width = 224
image_height = 224

model = tf.keras.models.load_model(model_path)


@app.route("/")
def hello_world():
    return {
        "message": model_path
    }


@app.route("/upload", methods=['POST'])
def upload():
    if 'image' not in request.files:
        message = "No file part"
        response = 400

        return {
            "response": response,
            "message": message,
        }

    file = request.files['image']
    if file.filename == '':
        message = "No selected file"
        response = 400

        return {
            "response": response,
            "message": message,
        }

    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], "image.jpg"))

        message = "Sukses upload data"
        response = 200

        return {
            "response": response,
            "message": message,
        }

    return {
        "response": "",
        "message": 0,
    }


@app.route("/predict")
def predict():
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(image_width, image_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    prediction = predictions[0][0]

    if prediction > 0.5:
        print("jernih")
        return {
            "predict": "jernih",
            "skor": float(prediction)
        }
    else:
        print("blur")
        return {
            "predict": "blur",
            "skor": float(prediction)
        }


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
