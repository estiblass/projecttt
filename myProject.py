from tensorflow.keras.models import Sequential
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array

dir_train = 'Train'
dir_test = 'Test'

def load_images(path):
    path, dirs, files = next(os.walk(path))
    file_count = len(files)
    # ניצור מסד נתונים טבלאי מהתמונות בתיקיית האימון תוך אבחנה בין הקטגוריות:
    filenames = os.listdir(path)
    categories = []
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


def convert_df_to_numpy(df, dir):
    list = []
    for i in df['filename']:
        image = load_img(dir + "\\" + i, target_size=(28, 28), color_mode="grayscale")
        input_arr = img_to_array(image)
        list.append(input_arr)
    X = np.array(list)
    return X


def confusionMatrix(y_test, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print(y_pred)
    print(y_test)
    confusion_matrix1=confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred, digits=4))
    ConfusionMatrixDisplay(confusion_matrix1).plot()
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
        X_train = X_train / 255
        X_test = X_test / 255
        return X_train, y_train, X_test, y_test, num_classes

    # בניית המודל
    @staticmethod
    def baseline_model(num_classes):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1),
                         activation='relu'))
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
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def fit_model(model, X_train, y_train, X_test, y_test):
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=1, mode='auto')
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=100, batch_size=64, verbose=2, callbacks=[es])
        model.save_weights("model_try5.h5")  # Save the model to a file
        model.save("model2.h5")
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



