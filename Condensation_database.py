import cv2
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import os


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
    background.save('nnnewww/' + path_dest)

# Run the function to resize all the images in the 'ori_dir' and save them to 'resized1' folder
newpath = r'nnnewww'
filename = os.listdir(newpath)
for item in tqdm(filename):
    file = os.path.join(newpath, item)
    resize_with_white_background(file, item)

