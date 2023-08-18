from flask import Flask , request , jsonify , render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
model = None

@app.route('/' , methods=["GET", "POST"])
def index():
    return render_template('index.html')

def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    return x

flowers_labels= ['roses',
                'daisy',
                'dandelion',
                'sunflowers',
                'tulips',]



@app.route("/predict" , methods=["POST","GET"])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file: #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path) #prepressing method
            
            class_prediction=model.predict(img) 
            classes_x=np.argmax(class_prediction)

            return jsonify({"predictions":flowers_labels[classes_x]})

        else:
            return jsonify({'message': 'No image uploaded'})
    
    
if __name__ == '__main__':
    model = tf.keras.models.load_model('my_model')
    print("app is running")
    
    app.run(debug=True)