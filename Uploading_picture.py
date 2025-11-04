from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import os
import cv2
import shutil
import numpy as np

def resize_with_white_background(path_ori, path_dest):
    img = Image.open(path_ori)
    img.thumbnail((28, 28), Image.LANCZOS)
    img_w, img_h = img.size
    background = Image.new('L', (28, 28))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
    background.paste(img, offset)
    background.save('Cut_image/' + path_dest)


def cut_image(image_path):
    args = Image.open(image_path)
    imge = image_path
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.equalizeHist(gray)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(thresh, 100, 200)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    distances = []
    boundes = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        boundes.append((x, y, w, h))
        sorted_boundes = sorted(boundes, key=lambda b: b[0])  # Sort based on the x-coordinate (b[0])
    # מוצא את הרווח בין קווי המתאר
    for i in range(len(contours) - 1):
        x, y, w, h = sorted_boundes[i]
        w1, w3, w2, w4 = sorted_boundes[i + 1]
        dist = w1 - (x + w)
        print(dist)
        cv2.rectangle(edges, (x, y), (x + w, y + h), (255, 0, 0), 2)
        distance = dist  # Calculate the distance between contour areas
        distances.append(distance)
    avg = np.average(distances)
    min_area = 0
    max_area = 500
    contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2], reverse=True)
    newpath = r'Cut_image'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        letter = img[y:y + h, x:x + w]
        cv2.imwrite('letter_{}.jpg'.format(i), letter)
        new_path = os.path.join(newpath, 'letter_{}.jpg'.format(i))
        shutil.move('letter_{}.jpg'.format(i), new_path)
    filename = os.listdir(newpath)
    for item in tqdm(filename):
        file = os.path.join(newpath, item)
    resize_with_white_background(file, item)
    return avg, distances

