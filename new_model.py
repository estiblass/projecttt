
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

model = load_model('model.h5')

def convert_df_to_numpy(dir,files):
    list = []
    for i in files:
        image = load_img(dir + "\\" + i, target_size=(28, 28), color_mode="grayscale")
        input_arr = img_to_array(image)
        list.append(input_arr)
    X = np.array(list)
    return X

def load_model(path,files):
    X_train = convert_df_to_numpy(path,files)
    X_train = X_train / 255
    return X_train

def get_max_prediction(prediction):
    max_value = np.max(prediction)
    max_index = np.argmax(prediction)
    return max_value, max_index

def convert_to_hebrew_letters(numbers):
    hebrew_letters = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ','ך', 'ל', 'מ','ם', 'נ','ן', 'ס', 'ע', 'פ','ף', 'צ','ץ', 'ק', 'ר', 'ש', 'ת']
    letters = [hebrew_letters[num] for num in numbers]
    return letters

def predict():
    path = 'Cut_image'
    files = os.listdir(path)
    arr_predict = []
    p = load_model(path,files)
    y_hat = model.predict(p)
    for i in y_hat:
        value ,index = get_max_prediction(i)
        arr_predict.append(index)
    letters=convert_to_hebrew_letters(arr_predict)
    print(letters)
    return letters