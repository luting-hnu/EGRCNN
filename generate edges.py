
import cv2
import glob
import os

#source path
image_path = '/home/luting/桌面/Bai/CD/dataset/train/'
image_list = glob.glob(image_path + 'label/*.png')
#target path
dir_name = '/home/luting/桌面/Bai/CD/dataset/train/edge/'

for i in range(len(image_list)):
    img = cv2.imread(image_list[i])
    canny = cv2.Canny(img, 50, 150)
    basename = os.path.basename(image_list[i])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(dir_name + basename, canny)
