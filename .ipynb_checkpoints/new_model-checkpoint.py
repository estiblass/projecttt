# מקבלת תמונה
# תסדר את התמונה
# baseline_model
# lode wheit לקובץ
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import myProject
from myProject import ModelTraining
path = 'Cut_image'
def send_to_new_model():
    imge = path
    model = load_model('model_try3.h5')
    X_train, y_train, X_test, y_test, num_classes=myProject.ModelTraining.load_model()
    myProject.ModelTraining.baseline_model(num_classes)
    prediction = model.predict(imge)
    print(prediction)
    # if prediction > 0.5:
    #     print("Waldo")
    # else:
    #     print("Not Waldo")
