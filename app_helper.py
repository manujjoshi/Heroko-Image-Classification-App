# Import the model and other libraries
from tensorflow.keras.applications.resnet50 import ResNet50 as myModel
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

from tensorflow.keras.preprocessing import image
import numpy as np

def get_classes(file_path):
    # Create an instance of 'myModel' imported above
    model = myModel(weights="imagenet")

    # Load image and preprocess it
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x= np.array([x])
    x = preprocess_input(x)

    # This is the inference time. Given an instance, it produces the predictions.
    preds = model.predict(x)
    predictions = decode_predictions(preds, top=3)[0]
    return predictions
