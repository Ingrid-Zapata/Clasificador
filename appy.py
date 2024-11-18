from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Cargar el modelo preentrenado
model = tf.keras.applications.MobileNetV2(weights='imagenet')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    
    # Decodificar la imagen
    image_data = data['image']
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocesar la imagen
    image = image.resize((224, 224))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:  # Si hay un canal alpha
        image_array = image_array[..., :3]

    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    # Hacer la predicción
    predictions = model.predict(image_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    # Formatear las respuestas
    results = [{
        'label': label,
        'probability': float(probability)
    } for (_, label, probability) in decoded_predictions]

    return jsonify(results)

if __name__ == '__main__':
    app.run(port=55000, debug=True)