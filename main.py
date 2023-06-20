import cv2
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File
from keras.models import load_model
from tensorflow import keras


app = FastAPI()

# Load the pre-trained model
model = load_model(r'C:\Users\isteb\Desktop\6thSem\ArtificialIntelligence\project\our_model.h5')

# Load the image and convert it to grayscale
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (224, 224))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, 224, 224, 1))
    return img_reshaped

def predict(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    # return prediction[0][0]
    # Convert the prediction to a human-readable label
    label = "signature" if prediction[0][0] >= 0.5 else "face"
    # print(label)
    return label

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict_signature(file: UploadFile = File(...)):
    # Save the uploaded file locally
    with open("temp_image.jpg", "wb") as f:
        f.write(file.file.read())

    # Perform prediction using the ML model
    prediction = predict("temp_image.jpg")

    # Remove the temporary image file
    os.remove("temp_image.jpg")

    # Return the prediction as the API response
    return {"prediction": prediction}
