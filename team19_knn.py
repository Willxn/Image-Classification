######################################################################
#                       Code running guide                           #
#      1. Modify the paths of the training set and test set          #
#      2. Modify the file format of the image to .png or .jpg        #
#      3. Modify train_flag and test_flag. True means running        #
#      4. To view the preprocessed image, set debug=True             #
#                                                                    #
######################################################################

import cv2
import csv
import numpy as np

###############################################
#                 Parameters                  #
###############################################

# Directory paths
#trainDatasetDirectory = './dataset/2024Simgs/'
testDatasetDirectory = './dataset/2024Simgs/'

# KNN parameters
k = 3
dimension = 3  # image channel
train = []
test = []

# Debug mode(Flags for displaying preprocessed image)
debug = False

# Flags for training and testing
train_flag = True
test_flag = True


##########################################
#            Image Processing            #
##########################################

def preprocess(image):
    # 增加亮度
    # 将图像转换为HSV
    value = 20
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 直接在V通道上增加亮度
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    # 将修改后的HSV图像转换回BGR格式
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 裁剪图像
    cropped_image = crop_image(image)

    return cropped_image

def crop_image(image):
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    exclusion_zone = 10
    gray[-int(h / 5):, :] = 0
    gray = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    gray = cv2.erode(gray, np.ones((8, 8), np.uint8))
    _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    valid_contours = [
        cnt for cnt in contours
        if cnt.shape[0] > 100
        and not (
            np.any(cnt[:, :, 1] > h - exclusion_zone)
            or np.any(cnt[:, :, 1] < exclusion_zone)
        )
    ]
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:3]

    mid_contour = min(
        valid_contours,
        key=lambda cnt: abs(np.mean(cnt[:, :, 1]) - h / 2),
        default=None
    )

    if mid_contour is not None:
        x, y, w, h = cv2.boundingRect(mid_contour)
        x1, y1, x2, y2 = max(x - 10, 0), max(y - 10, 0), min(x + w + 10, image.shape[1]), min(y + h + 10, image.shape[0])
        cropped = image[y1:y2, x1:x2]
    else:
        cropped = np.zeros((30, 30, 3), dtype='uint8')

    cropped = cv2.resize(cropped, (40, 40), interpolation=cv2.INTER_LINEAR)
    return cropped


##################################################
#                     Train                      #
##################################################

if train_flag:
    data_path = f'{trainDatasetDirectory}labels.txt'
    with open(data_path, 'r') as file:
        data_reader = csv.reader(file)
        data_lines = [line for line in data_reader]

    training_images = []
    for data_line in data_lines:
        image_path = f"{trainDatasetDirectory}{data_line[0]}.png"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        preprocessed_image = preprocess(image)
        training_images.append(preprocessed_image)

        if debug:
            cv2.imshow("Processed Image", preprocessed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    flattened_images = np.array(training_images).reshape(len(training_images), -1).astype(np.float32)
    labels = np.array([int(item[1]) for item in data_lines])

    model = cv2.ml.KNearest_create()
    model.train(flattened_images, cv2.ml.ROW_SAMPLE, labels)
    model.save("Team19_Model.xml")


##################################################
#                      Test                      #
##################################################

if test_flag:
    # Load the pre-trained KNN model
    knn_model = cv2.ml.KNearest_create().load("Team19_Model.xml")
    with open(f'{testDatasetDirectory}labels.txt', 'r') as file:
        data_reader = csv.reader(file)
        test_data = list(data_reader)

    accuracy_count = 0.0
    num_classes = 6
    confusion_matrix = np.zeros((num_classes, num_classes))

    for data in test_data:
        img_file = f'{testDatasetDirectory}{data[0]}.png'
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        preprocessed_img = preprocess(img)

        if debug:
            cv2.imshow("Test Image", img)
            cv2.imshow("Preprocessed Test Image", preprocessed_img)
            if cv2.waitKey() == 27:
                break

        img_vector = preprocessed_img.flatten().reshape(1, -1).astype(np.float32)
        actual_label = int(data[1])

        _, result, _, _ = knn_model.findNearest(img_vector, k)
        predicted = int(result[0, 0])  # Extract the first element to get the scalar

        if actual_label == predicted:
            accuracy_count += 1
            confusion_matrix[predicted, predicted] += 1
        else:
            confusion_matrix[actual_label, predicted] += 1
            # print(line[0] + " Wrong,", test_label, "classified as", ret)
            # print("\tneighbours:", neighbours)
            # print("\tdistances:", dist)

    print("\nTest accuracy:", accuracy_count / len(test_data))
    print("\nConfusion matrix:")
    print(confusion_matrix)
