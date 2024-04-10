import cv2
import numpy as np

def increase_brightness(img, value=20):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    brightened_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return brightened_img


import cv2
import numpy as np


import cv2
import numpy as np

import cv2
import numpy as np


import cv2
import numpy as np

def crop(image):
    img_h, img_w, _ = image.shape
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_img = cv2.bitwise_not(gray_img)

    inverted_img[-int(img_h / 5):, :] = 0
    dilated_img = cv2.dilate(inverted_img, np.ones((7, 7), np.uint8))
    eroded_img = cv2.erode(dilated_img, np.ones((8, 8), np.uint8))

    _, binary_img = cv2.threshold(eroded_img, 130, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 200 and not (np.any(contour[:, :, 1] > img_h - 10) or np.any(contour[:, :, 1] < 10))]
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)

    closest_contour = None
    if valid_contours:
        closest_contour = min(valid_contours, key=lambda contour: abs(np.mean(contour[:, :, 1]) - img_h / 2))

    if closest_contour is not None:
        x, y, w, h = cv2.boundingRect(closest_contour)
        cropped_img = image[y - 10 if y - 10 > 0 else 0:y + h + 10 if y + h + 10 < img_h else img_h, x - 10 if x - 10 > 0 else 0:x + w + 10 if x + w + 10 < img_w else img_w]
    else:
        cropped_img = np.zeros((30, 30, 3), dtype='uint8')

    resized_img = cv2.resize(cropped_img, (40, 40), interpolation=cv2.INTER_LINEAR)
    return resized_img



def main(image_path, value=20):
    # 加载图像
    img = cv2.imread(image_path)
    if img is None:
        print("Error: 图像未找到或无法读取。")
        return

    # 增加亮度
    brightened_img = increase_brightness(img, value)
    cropped_img = crop(brightened_img)

    # 显示原图和增亮后的图像
    cv2.imshow("Original Image", img)
    cv2.imshow("Brightened Image", cropped_img)

    # 等待按键后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用实例：增亮'/path/to/your/image.jpg'
main('./dataset/2023Fimgs/1.jpg')
