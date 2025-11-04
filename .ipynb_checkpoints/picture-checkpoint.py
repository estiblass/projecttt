import os
import imutils
from PIL import Image
from keras.models import  load_model
from imutils.contours import sort_contours
import pytesseract
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

args = Image.open('w.png')
imge = 'w.png'

args.show()
# img = cv2.imread('D:/ww.jpg')
# #Alternatively: can be skipped if you have a Blackwhite image
# grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# grey, img_bin = cv2.threshold(grey,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# grey = cv2.bitwise_not(img_bin)
# kernel = np.ones((2, 1), np.uint8)
# img = cv2.erode(grey, kernel, iterations=1)
# img = cv2.dilate(img, kernel, iterations=1)
# out_below = pytesseract.image_to_string(img)
# print("OUTPUT:", out_below)


# load the input image from disk, convert it to grayscale, and blur
# it to reduce noise
#used to read an image from a file
image =cv2.imread(imge)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# print(gray)

print("@@@@@@@@@@@@@@")
# print(image)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(blurred, 30, 150)
# המרת מטריצה להצגת התמונהarray = edged
print(type(array))
print(array.shape)
array = np.reshape(array, (48, 233))
print(array.shape)
print(array)
data = Image.fromarray(array)
data.show()
cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]
print("@@@@@@@@@@@@@@@@@@")
# print(cnts[0])
# print(type(cnts))
chars = []



print(len(cnts))
# loop over the contours
for c in cnts:
    print(c)
    print("#########")
    # compute the bounding box of the contour
    (x, y, w, h) =cv2.boundingRect(c)
    # filter out bounding boxes, ensuring they are neither too small
    # nor too large
    if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
        # extract the character and threshold it to make the character
        # appear as *white* (foreground) on a *black* background, then
        # grab the width and height of the thresholded image
        roi = gray[y:y + h, x:x + w]
        thresh =cv2.threshold(roi, 0, 255,cv2
                               .THRESH_BINARY_INV |cv2.THRESH_OTSU)[1]
        (tH, tW) = thresh.shape
        # if the width is greater than the height, resize along the
        # width dimension
        if tW > tH:
            thresh = imutils.resize(thresh, width=32)
        # otherwise, resize along the height
        else:
            thresh = imutils.resize(thresh, height=32)

        # re-grab the image dimensions (now that its been resized)
        # and then determine how much we need to pad the width and
        # height such that our image will be 32x32
        (tH, tW) = thresh.shape
        dX = int(max(0, 32 - tW) / 2.0)
        dY = int(max(0, 32 - tH) / 2.0)
        # pad the image and force 32x32 dimensions
        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                    left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))
        padded = cv2.resize(padded, (32, 32))
        # prepare the padded image for classification via our
        # handwriting OCR model
        padded = padded.astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)

        # update our list of characters that will be OCR'd
        chars.append((padded, (x, y, w, h)))
        print(padded)
        array = padded
        print(type(array))
        print(array.shape)
        row = np.prod(array.shape)
        print(row)
        array = np.reshape(array, (int(row / 2), 2))
        print(array.shape)
        print(array)
        data = Image.fromarray(array)
        data.show()
