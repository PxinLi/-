import cv2
import numpy as np
import glob
import os
import svm

# 读取图片，因为路径中含有中文字符，所以特殊处理
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img


def seg_plate_to_char(img):

    # 定义二值化阈值,可根据应用进行调整
    binary_threshold = 130
    # 1、读取图片，并做灰度处理
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', img_gray)
    # cv2.waitKey(0)

    # 2、将灰度图二值化，设定阀值为100
    img_thre = img_gray
    cv2.threshold(img_gray, binary_threshold, 255, cv2.THRESH_BINARY, img_thre)
    # cv2.imshow('threshold', img_thre)
    # cv2.waitKey(0)

    # 3、分割字符
    white = []  # 记录每一列的白色像素总和
    black = []  # 记录每一列的黑色像素总和
    height = img_thre.shape[0]
    width = img_thre.shape[1]

    # 循环计算每一列的黑白色像素总和

    white_max = 0  # 仅保存每列，取列中白色最多的像素总数
    black_max = 0  # 仅保存每列，取列中黑色最多的像素总数

    for i in range(width):
        w_count = 0  # 这一列白色总数
        b_count = 0  # 这一列黑色总数
        for j in range(height):
            if img_thre[j][i] == 255:
                w_count += 1
            else:
                b_count += 1
        white.append(w_count)
        white_max = max(white_max, w_count)
        black_max = max(black_max, b_count)
        black.append(b_count)

    n = 1
    segmentation_spacing = 0.90
    seg_img = []
    while n < width - 1:
        n += 1
        if white[n] > (1 - segmentation_spacing) * white_max :
            start = n
            end_ = start + 1
            for m in range(start + 1, width - 1):
                if black[m] > segmentation_spacing * black_max :
                    end_ = m
                    break
            end = end_
            n = end
            if end - start > 5:
                h1 = int(0.15*height)
                h2 = int(0.85 * height)
                cj = img_thre[h1:h2, start-3:end+3]
                cjh, cjw = cj.shape[0], cj.shape[1]
                if cjh < cjw:
                    cj_s = cv2.copyMakeBorder(cj, int((cjw-cjh)/2), int((cjw-cjh)/2), 0, 0, cv2.BORDER_CONSTANT)
                else:
                    cj_s = cv2.copyMakeBorder(cj, 0, 0, int((cjh-cjw)/2), int((cjh-cjw)/2), cv2.BORDER_CONSTANT)
                cj_s_20 = cv2.resize(cj_s, (20, 20))

                seg_img.append(cj_s_20)

    return seg_img


if __name__ == '__main__':
    char_svm = svm.SVM_char()
    char_svm.train()

    crop_file = glob.glob('./crop_plate/*.jpg')
    if not os.path.exists('./crop_plate_seg'):
        os.mkdir('./crop_plate_seg')
    for crop_name in crop_file:
        img_name_pre = crop_name.split("\\")[1]
        label_str = img_name_pre[1:7]
        label_list = [ord(i) for i in label_str]
        image = cv_imread(crop_name)
        seg_img = seg_plate_to_char(image)
        if len(seg_img) == 7:
            y_pred = char_svm.predict(seg_img[1:])
            print(y_pred)
            print(label_list)