import cv2
import numpy as np
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.applications import mobilenet
from keras.models import load_model
from PIL import Image, ImageOps


class Classifier:


    def __init__(self, modelPath, labelsPath=None):
        self.model_path = modelPath
        np.set_printoptions(suppress=True) 

        self.model = load_model(self.model_path)


        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        self.labels_path = labelsPath

        if self.labels_path:
            label_file = open(self.labels_path, "r")
            self.list_labels = [line.strip() for line in label_file]
            label_file.close()
        else:
            print("No Labels Found")

    def getPrediction(self, img, draw=True, pos=(50, 50), scale=2, color=(0, 255, 0)):

        imgS = cv2.resize(img, (224, 224))
        image=np.expand_dims(imgS,axis=0)
        image_mobilenet=mobilenet.preprocess_input(image)


        self.data[0] = image_mobilenet

        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        if draw and self.labels_path:
            cv2.putText(img, str(self.list_labels[indexVal]), pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

        return list(prediction[0]), indexVal