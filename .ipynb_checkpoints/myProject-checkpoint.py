import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.image as mpimg
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import itertools
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.saving import hdf5_format
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
# importing tensorflow for doing ML
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from sklearn.metrics import confusion_matrix ,accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn import model_selection, datasets
from sklearn.tree import DecisionTreeClassifier

# import sklearn.external.joblib as extjoblib
import joblib
import pickle

# from tensorflow.python.keras.applications import VGG16
# from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.optimizers import Adam, RMSprop
# https://reshetech.co.il/machine-learning-tutorials/classify_images_kagel_dataset_vgg16_pretrained_model

dir_train = 'Train'
dir_test = 'Test'


# נבדוק כמה תמונות יש בכל תיקייה:
# path, dirs, files = next(os.walk(dir_train))
# file_count_train = len(files)
# print(file_count_train)

def load_images(path):
    path, dirs, files = next(os.walk(path))
    file_count = len(files)
    # ניצור מסד נתונים טבלאי מהתמונות בתיקיית האימון תוך אבחנה בין הקטגוריות:
    filenames = os.listdir(path)
    categories = []
    # לשנות למשתנה
    df = ([[file_count], [2]])
    for filename in filenames:
        category = filename.split('_')[0]
        categories.append(category)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    return df


df_train = load_images(dir_train)
df_test = load_images(dir_test)


classNames = ['0', '1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15', '16', '17', '18', '19', '20',
             '21', '22', '23', '24', '25', '26']


def convert_df_to_numpy(df, dir):
    list = []
    for i in df['filename']:
        image = load_img(dir + "\\" + i, target_size=(28, 28), color_mode="grayscale")
        input_arr = img_to_array(image)
        list.append(input_arr)
    X = np.array(list)
    return X

#@staticmethod
def confusionMatrix(y_test, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print(y_pred)
    print(y_test)
    confusion_matrix1=confusion_matrix(y_test, y_pred)#, labels=classNames)
    #confusion_matrix1 = confusion_matrix([24,24,24,10,0,0,2,1,5,0,0,1,1,2,3,4], [24,23,24,10,0,0,2,1,4,0,4,1,1,2,3,4])#, labels=classNames)
    print(classification_report(y_test, y_pred, digits=4))
    ConfusionMatrixDisplay(confusion_matrix1).plot()#, display_labels=classNames).plot()
    plt.show()
# טעינת התמונות לרשת
class ModelTraining:
    @staticmethod
    def load_model():
        y_train = to_categorical(df_train['category'])
        y_test = to_categorical(df_test['category'])
        num_classes = y_test.shape[1]
        X_train = convert_df_to_numpy(df_train, dir_train)
        X_test = convert_df_to_numpy(df_test, dir_test)
        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255
        return X_train, y_train, X_test, y_test, num_classes


    # בניית המודל
    @staticmethod
    def baseline_model(num_classes):
        # Create a layer type model
        model = Sequential()
        # convolution layer: 32 filters, 3*3, use with relu for activation function
        model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1),
                         activation='relu'))
        # [add this layer to decrease the loss]
        model.add(Conv2D(64, (3, 3),
                         activation='relu'))
        model.add(Conv2D(128, (3, 3),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3),
                         activation='relu'))
        model.add(Conv2D(256, (3, 3),
                         activation='relu'))
        model.add(Conv2D(128, (3, 3),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # ignore from 20% from the noironim
        model.add(Dropout(0.5))
        # we have 3 chanels. flat them to one long vector
        model.add(Flatten())
        # another noirinim layer (with activation function)
        model.add(Dense(128, activation='relu'))
        # output
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        # model.summary()
        return model

    @staticmethod
    def fit_model(model, X_train, y_train, X_test, y_test):
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=1, mode='auto')
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=100, batch_size=64, verbose=2, callbacks=[es])
        # model.fit(X_train, y_train, validation_data=(X_test, y_test),
        #           epochs=100, batch_size=64, verbose=2)
        # model.save("model_try3.h5")
        # tf.saved_model("model_try3.h5")
        model.save_weights("model_try3.h5")  # Save the model to a file
        # filename = "Completed_model.joblib"
        # joblib.dump(model, filename)
        yhat=model.predict(X_test)
        print(yhat)
        print(y_test)
        confusionMatrix(y_test,yhat)



a = ModelTraining
X_train, y_train, X_test, y_test, num_classes = a.load_model()

model = a.baseline_model(num_classes)
print(y_test.shape)
print(y_train.shape)
a.fit_model(model, X_train, y_train, X_test, y_test)



