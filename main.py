from flask import Flask, request
from flask_cors import CORS
import tensorflow as tf
import os
import numpy as np

app = Flask(__name__)
CORS(app)

root_path = os.path.dirname(__file__)
model_path = os.path.join(root_path, "model.h5")

image_width = 224
image_height = 224

model = tf.keras.models.load_model(model_path)


def predict(filename):
    img = tf.keras.preprocessing.image.load_img(filename, target_size=(image_width, image_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    prediction = predictions[0][0]

    if prediction > 0.5:
        print("jernih")
    else:
        print("blur")


@app.route("/")
def hello_world():
    return {
        "message": model_path
    }


@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    predict(data)
    return {
        "response": 'sukses',
        # "summary": summary,
    }


if __name__ == "__main__":
    app.run()
