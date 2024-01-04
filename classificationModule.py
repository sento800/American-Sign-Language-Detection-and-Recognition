import cv2
import numpy as np
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.applications import mobilenet
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps


class Classifier:
    """
    Classifier class that handles image classification using a pre-trained Keras model.
    """

    def __init__(self, modelPath, labelsPath=None):
        """
        Khởi tạo Trình phân loại với mô hình và nhãn.

        :param modelPath: str, đường dẫn đến mô hình Keras 
        :param labelsPath: str, đường dẫn đến tệp nhãn (tùy chọn)
        """
        self.model_path = modelPath
        np.set_printoptions(suppress=True) 

        # Tải mô hình Keras
        self.model = load_model(self.model_path)

        # Tạo một mảng NumPy có hình dạng phù hợp để đưa vào mô hình Keras
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    def getPrediction(self, img, draw=True, pos=(50, 50), scale=2, color=(0, 255, 0)):
        """
        Phân loại hình ảnh và tùy ý vẽ kết quả lên hình ảnh. 

        :param img: hình ảnh để phân loại 
        :param draw: có vẽ dự đoán trên hình ảnh hay không 
        :param pos: vị trí nơi vẽ văn bản 
        :param tỉ lệ: tỷ lệ phông chữ 
        :param color: màu văn bản 
        :return: danh sách dự đoán, chỉ mục của dự đoán có khả năng nhất
        """
        # Resize and normalize ảnh
        imgS = cv2.resize(img, (224, 224))
        image=np.expand_dims(imgS,axis=0)
        image_mobilenet=mobilenet.preprocess_input(image)

        self.data[0] = image_mobilenet

        # Run inference
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)

        return list(prediction[0]), indexVal