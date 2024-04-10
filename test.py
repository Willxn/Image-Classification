import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Function to load images and their labels

import cv2
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blur, 50, 200)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and aspect ratio
    sign_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        # These thresholds can be adjusted to fit the specific size and aspect ratio of your signs
        if area > 1000 and 0.75 < aspect_ratio < 1.33:
            sign_contours.append(cnt)

    # Assuming the largest contour after filtering is the sign
    if sign_contours:
        largest_contour = max(sign_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        sign_roi = image[y:y + h, x:x + w]
        return sign_roi
    else:
        return image



def preprocess_and_display_image(image, display=False, save=False, save_path=None):
    # 进行预处理
    preprocessed_image = preprocess_image(image)  # 假设这个函数已经定义

    # 如果需要显示图像
    if display:
        cv2.imshow('Preprocessed Image', preprocessed_image)
        cv2.waitKey(0)  # 等待按键后再继续
        cv2.destroyAllWindows()

    # 如果需要保存图像
    if save and save_path is not None:
        cv2.imwrite(save_path, preprocessed_image)

    return preprocessed_image



def load_images_and_labels(images_folder, labels_file_path):
    labels = {}
    with open(labels_file_path, 'r') as file:
        for line in file:
            # 使用逗号分隔文件名和标签
            image_name, label = line.strip().split(', ')
            # 删除文件名中的可能的空格，并添加.png后缀
            image_name = image_name.strip() + '.png'
            labels[image_name] = int(label.strip())
    images = [(cv2.imread(os.path.join(images_folder, img_name)), lbl) for img_name, lbl in labels.items()]
    return images


# Function to preprocess images and extract features
def extract_features(image):
    # Convert to HSV color space for color invariance
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Extract color histogram in HSV space and flatten into a feature vector
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Function to create the feature matrix and label vector
def create_feature_matrix_and_labels(images_with_labels):
    feature_matrix = []
    label_vector = []
    for image, label in images_with_labels:
        if image is not None:  # Only if the image was correctly loaded
            # 应用预处理
            preprocessed_image = preprocess_image(image)
            features = extract_features(preprocessed_image)
            feature_matrix.append(features)
            label_vector.append(label)
    return np.array(feature_matrix), np.array(label_vector)

# Paths
images_folder = '/Users/will/Documents/Spring24/ECE7785/7785_Lab6/2022Fimgs/'
train_labels_file_path = os.path.join(images_folder, 'train.txt')
test_labels_file_path = os.path.join(images_folder, 'test.txt')

# Load training and testing data
training_data = load_images_and_labels(images_folder, train_labels_file_path)
testing_data = load_images_and_labels(images_folder, test_labels_file_path)

# # 在处理图像的循环中调用该函数
# for image_path, label in training_data:
#     preprocessed_image = preprocess_and_display_image(
#         image_path,
#         display=True,  # 设置为True以显示图像
#         save=True,  # 设置为True以保存图像
#         save_path='/Users/will/Desktop/pics/' + str(label) + '.png'  # 确保label被转换为字符串
#   # 指定保存路径
#     )


print(f"Loaded {len(training_data)} training images.")
print(f"Loaded {len(testing_data)} testing images.")

# Extract features and labels
X_train, y_train = create_feature_matrix_and_labels(training_data)
X_test, y_test = create_feature_matrix_and_labels(testing_data)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Evaluate the classifier
train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# 创建一个SVM模型并使用Pipeline包括预处理步骤
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 加入标准化步骤
    ('svm', SVC())
])

# 更新参数网格，考虑Pipeline
param_grid = {
    'svm__C': [0.1, 1, 10, 100, 1000],
    'svm__gamma': [0.001, 0.01, 0.1, 1, 'scale'],
    'svm__kernel': ['rbf', 'linear'],
    'svm__class_weight': [None, 'balanced']  # 处理不平衡数据
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(pipeline, param_grid, cv=10, verbose=2, n_jobs=-1)  # 使用更多的CPU核心

# 在训练数据上执行网格搜索
grid_search.fit(X_train, y_train)

# 查看最佳参数
print("Best parameters:", grid_search.best_params_)

# 使用最佳参数的模型进行预测和评估
best_pipeline = grid_search.best_estimator_
train_accuracy = best_pipeline.score(X_train, y_train)
test_accuracy = best_pipeline.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

