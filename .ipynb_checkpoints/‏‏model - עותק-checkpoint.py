import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# importing tensorflow for doing ML
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout

from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img,img_to_array


# from tensorflow.python.keras.applications import VGG16
# from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.optimizers import Adam, RMSprop
# https://reshetech.co.il/machine-learning-tutorials/classify_images_kagel_dataset_vgg16_pretrained_model

dir_train = r'newDatabase/TRAIN'
dir_test = r'newDatabase/TEST'

# נבדוק כמה תמונות יש בכל תיקייה:
path, dirs, files = next(os.walk(dir_train))
file_count_train = len(files)
print(file_count_train)

path, dirs, files = next(os.walk(dir_test))
file_count_test = len(files)
print(file_count_test)

# ניצור מסד נתונים טבלאי מהתמונות בתיקיית האימון תוך אבחנה בין הקטגוריות:
filenames = os.listdir(dir_test)

categories = []
# לשנות למשתנה
df = ([[file_count_test], [2]])
for filename in filenames:
    category = filename.split('_')[0]
    categories.append(category)
    # for i in range(28):
    #     if category == i:
    #         categories.append(i)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

# נסקור את מסד הנתונים
df.head()


# for i in range(28):
#    print('number of %i: %d' % len(df[df.category == i]))


# visualize the images
# import matplotlib.image as mpimg


# נציג 4 תמונות באקראי לצד הקטגוריה אליהם הם משתייכות באמצעות הפונקציה:
# Create figure with 2x2 sub-plots.
def plot_images(images, labels):
    fig, axes = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image
        ax.imshow(mpimg.imread(images[i]))

        # Plot label
        ax.set_xlabel('Label : %s' % labels[i])

    plt.show()


# נשלוף 4 רשומות אקראיות מתוך מסד הנתונים ונשתמש בפונקציה לעיל כדי להציג אותם:
# מציג תמונה מהדטהביס באקראי וכותב מתחת את האות צריך לתקן שיהיה כתוב במקום cat  את האות
img_paths = []
img_labels = []
for i in range(4):
    # pick 4 random ids from the dataset
    rand_id = np.random.randint(0, file_count_test)

    # get the img path from the id
    filename = df.loc[rand_id, 'filename']
    path = os.path.join(dir_test, filename)
    img_paths.append(path)

    # get the img label from the id
    img_label = df.loc[rand_id, 'category']
    img_labels.append(img_label)

plot_images(img_paths, img_labels)


# print(img_label)
# for g in range(28):
#     if img_label == g:
#         img_labels.append(g)

# https://reshetech.co.il/machine-learning-tutorials/directory_structure_for_classify_images_kagel_dataset_vgg16_model

# עיבוד התמונות לגודל מתאים


# resize with white background instead of missing pixels
def resize_with_white_background(path_ori, path_dest):
    img = Image.open(path_ori)

    # resize and keep the aspect ratio
    img.thumbnail((28, 28), Image.LANCZOS)

    # add the white background
    img_w, img_h = img.size
    background = Image.new('L', (28, 28))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset)
    background.save('resized1/' + path_dest)

ori_dir = r'newDatabase/TEST'
# run the function to resize all the images in the 'ori_dir'
# for item in tqdm(df['filename']):
#    file = ori_dir + '/' + item
#    resize_with_white_background(file, item)
# img_paths_resized = []


classNames = ['0', '1', '3','4','5','6','7','2','2','2','2','2','2']

def getLabel(file_path):
    # Convert the path to a list of path components
    fileName = tf.strings.split(file_path, os.path.sep)[-1]
    # get label name from filename
    className = tf.strings.split(fileName, '_')[0]
    # get one_hot vector boolean
    one_hot = className == classNames
    # cast vector type to integer
    return tf.cast(one_hot, dtype=tf.int8, name=None)

def getImage(file_path):
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_bmp(img, channels=3)
    # cast tf.Tensor type to uint8
    return tf.cast(img, dtype=tf.uint8, name=None)

def process_path(file_path):
    label = getLabel(file_path)
    img = getImage(file_path)
    return img, label

path = r'Train'
ds = df.data.Dataset.list_files(path)
ds = ds.map(process_path)
print (ds)

# טעינת התמונות לרשת
class ModelTraining:
    @staticmethod
    def load_model():
        # # # create generator
        # datagen = ImageDataGenerator()
        # # # load and iterate training dataset
        # train_it = datagen.flow_from_directory(
        #      r'Train',
        #      class_mode='categorical',
        #      color_mode="grayscale",
        #      batch_size=6000,
        #      target_size=(28, 28))
        # # load and iterate test dataset
        # test_it = datagen.flow_from_directory(
        #      r'Test',
        #      class_mode='categorical',
        #      batch_size=6000,
        #      target_size=(28, 28), color_mode="grayscale")
        # # # confirm the iterator works
        # #X_train, y_train = train_it.next()
        #train df x
        for i in range(file_count_train):
            X_train=[]
            image_train = load_img(r'Train\0_1.png',target_size=(28, 28), color_mode="grayscale")
            input_arr = img_to_array(image_train)
            X_train.append(input_arr)
        print(X_train)
        print(type(X_train))
       #test df x
        for i in range(file_count_test):
            X_test = []
            image_test = load_img(r'Test\0_1.png', target_size=(28, 28), color_mode="grayscale")
            input_arr = img_to_array(image_test)
            X_test.append(input_arr)
        print(X_test)
        print(type(X_test))

        # X_test, y_test = test_it.next()
        X_train = np.array(X_train)
        print(X_train.shape)
        # filenames = os.listdir('Test')
        y_train = to_categorical(df['category'])
        y_test = to_categorical(df['category'])

        num_classes = y_test.shape[1]

        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255
        num_classes = y_test.shape[1]
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
        model.add(Conv2D(52, (3, 3), input_shape=(28, 28, 1),
                     activation='relu'))
        model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1),
                     activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(90, (3, 3), input_shape=(28, 28, 1),
                     activation='relu'))
        model.add(Conv2D(95, (3, 3), input_shape=(28, 28, 1),
                     activation='relu'))
        model.add(Conv2D(98, (3, 3), input_shape=(28, 28, 1),
                     activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # ignore from 20% from the noironim
        model.add(Dropout(0.2))

        # we have 3 chanels. flat them to one long vector
        model.add(Flatten())
        # another noirinim layer (with activation function)
        model.add(Dense(128, activation='relu'))
        # output
        model.add(Dense(28, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model
a = ModelTraining
b=a.load_model()

a.baseline_model(b.y_test.shape[1])

@staticmethod
def fit_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=25, batch_size=64, verbose=2)
    model.save("model_try3.h5")
