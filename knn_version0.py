#########################################################################
#                              代码运行指南                               #
#     1. 修改21，22行的训练集和测试集的路径                                  #
#     2. 根据图片的格式修改对应的训练部分和测试部分为.png或.jpg                 #
#     3. 修改 59 60 行的 train_flag 和 test_flag. True是运行False是不运行   #
#     4. 如果想要查看预处理后的图像可以让 56 行的debug=True，按任意键看下一张图   #
#                                                                        #
##########################################################################

import platform
import cv2
import csv
import numpy as np
import time

###############################################
#                 Parameters                  #
###############################################

# Directory paths
trainDirectory = './dataset/2023Fimgs/'
testDirectory = './dataset/2022Fheldout/'

# KNN parameters
k = 5
dimension = 3  # image channel
train = []
test = []

# Image processing parameters
rgb = 1  # cv.imread mode

morphKernel = 5  # Closing and Opening Kernel Size
maxObjects = 1  # Max number of object to detect
minObjectArea = 300  # Min number of pixels for an object to be recognized

WHITE = [255, 255, 255]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]

raw_h = 308
raw_w = 410

# Resize parameters
resize_h_ratio = 0.1
resize_w_ratio = 0.1
resize_h = int(raw_h * resize_h_ratio)
resize_w = int(raw_w * resize_w_ratio)

# RGB filter parameters
light_thresh = 60
contrast = 30

# Debug mode(Flags for displaying preprocessed image)
debug = False

# Flags for training and testing
train_flag = False
test_flag = True

# Version check
print("Your python version:", platform.python_version())
print("Minimum opencv version: 4.2.0")
print("Your opencv version:", cv2.__version__)


##########################################
#            Image Processing            #
##########################################

def preprocess(image):
    # 增加亮度
    image = increase_brightness(image)

    # 裁剪图像
    cropped_image = crop(image)

    return cropped_image

def increase_brightness(img, value=20):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def crop(image):
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = ~gray
    border = 10
    gray[-int(h / 5):, :] = 0
    gray = cv2.dilate(gray, np.ones((7, 7)))
    gray = cv2.erode(gray, np.ones((8, 8)))
    ret, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    pot_contours = [contour for contour in contours if len(contour) > 200 and not (np.any(contour.reshape(-1, 2)[:, 1] > h - border) or np.any(contour.reshape(-1, 2)[:, 1] < border))]
    sorted_contours = sorted(pot_contours, key=cv2.contourArea, reverse=True)
    largest = sorted_contours[:3]

    closest_contour = min(largest, key=lambda contour: abs(np.mean(contour.reshape(-1, 2)[:, 1]) - h / 2), default=None)
    if closest_contour is None:
        cropped = np.zeros((30, 30, 3), dtype='uint8')
    else:
        c = closest_contour.reshape(-1, 2).T
        corner1 = np.max([np.min(c, axis=1) - 10, [0, 0]], axis=0)
        corner2 = np.min([np.max(c, axis=1) + 10, [w - 1, h - 1]], axis=0)
        cropped = image[int(corner1[1]):int(corner2[1]), int(corner1[0]):int(corner2[0])]

    cropped = cv2.resize(cropped, (30, 30), interpolation=cv2.INTER_LINEAR)
    return cropped


##################################################
#                     Train                      #
##################################################

if train_flag:
    with open(trainDirectory + 'train.txt', 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    for line in lines:
        img = preprocess(cv2.imread(trainDirectory + line[0] + ".jpg", rgb))
        train.append(img)

        if debug:
            cv2.imshow("preprocessed image", img)
            cv2.waitKey()
            cv2.destroyWindow("preprocessed image")

    train_data = np.array(train).reshape(len(train), -1).astype(np.float32)
    train_labels = np.array([int(line[1]) for line in lines])

    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
    knn.save("knnModel.xml")

################################################
#                    Test                      #
################################################

if test_flag:
    tic = time.time()
    knn_test = cv2.ml.KNearest_create()
    knn_test = knn_test.load("knnModel.xml")
    with open(testDirectory + 'labels.txt', 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    correct = 0.0
    confusion_matrix = np.zeros((6, 6))

    for line in lines:
        original_img = cv2.imread(testDirectory + line[0] + ".png", rgb)
        test_img = preprocess(original_img)

        if debug:
            cv2.imshow("Original Image", original_img)
            cv2.imshow("Image Resized", test_img)
            key = cv2.waitKey()
            if key == 27:  # Esc key to stop
                break

        test_img = test_img.flatten().reshape(1, -1).astype(np.float32)
        test_label = int(line[1])

        ret, results, neighbours, dist = knn_test.findNearest(test_img, k)

        if test_label == ret:
            print(line[0] + " Correct,", ret)
            correct += 1
            confusion_matrix[int(ret)][int(ret)] += 1
        else:
            confusion_matrix[test_label][int(ret)] += 1
            print(line[0] + " Wrong,", test_label, "classified as", ret)
            print("\tneighbours:", neighbours)
            print("\tdistances:", dist)

    toc = time.time()
    print("\nRun Time:", toc - tic)
    print("\nTotal accuracy:", correct / len(lines))
    avg_T = (toc - tic) / len(lines)
    print("Avg. Prediction time spent per img: %2.2f s" % avg_T)
    print(confusion_matrix)
