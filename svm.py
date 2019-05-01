import numpy as np
from sklearn.svm import SVC
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class SVM_char(object):
    def __init__(self, C=1, gamma='auto'):
        self.model = SVC(C=C, gamma=gamma)
        self.feature = self.define_feature()
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()

    def define_feature(self):
        winSize = (20, 20)
        blockSize = (8, 8)
        blockStride = (4, 4)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradient = True
        # HOG特征描述器
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                                cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection,
                                nlevels, signedGradient)
        return hog

    def load_data(self, root='./train/chars2'):
        chars_train = []
        chars_label = []

        for root, dirs, files in os.walk(root):
            if len(os.path.basename(root)) > 1:
                continue
            root_int = ord(os.path.basename(root))
            for filename in files:
                filepath = os.path.join(root, filename)
                digit_img = cv2.imread(filepath)
                digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                chars_train.append(digit_img)
                chars_label.append(root_int)
        hog_descriptors = []
        for img in chars_train:
            hog_descriptors.append(self.feature.compute(img))
        hog_descriptors = np.squeeze(hog_descriptors)
        X_train, X_test, y_train, y_test = train_test_split(
            hog_descriptors, chars_label, test_size=0.1, random_state=1)
        return X_train, X_test, y_train, y_test

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def val(self):
        y_pred = self.model.predict(self.X_test)
        print(confusion_matrix(self.y_test, y_pred))
        print('*' * 35)
        print("Accuracy:   ", accuracy_score(self.y_test, y_pred))

    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)

    def predict(self, imgs):
        X=[]
        for img in imgs:
            X.append(self.feature.compute(img))
        X = np.squeeze(X)
        y_pred = self.model.predict(X)
        return y_pred



if __name__ == '__main__':
    svm = SVM_char()
    svm.train()
    svm.val()