from flask import Flask , request , jsonify , render_template
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import load_img
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

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
        if file: 
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