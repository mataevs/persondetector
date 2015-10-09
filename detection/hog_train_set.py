
import cv2
import utils
from skimage.feature import hog
import cPickle
import numpy
from sklearn.externals import joblib
from sklearn import svm

def get_set(metadataFile, classType):
    set = []

    with open(metadataFile, "r") as f:
        entries = f.readlines()

    for entry in entries:
        entry = entry.split()
        filePath = entry[0]
        x, y, scale = int(entry[1]), int(entry[2]), float(entry[3])

        img = cv2.imread(filePath)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_crop = img_gray[y:y+128, x:x+64]

        hog_gray = hog(img_gray_crop, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualise=False)

        prevFilePath = utils.get_prev_img(filePath)


        prev_img = cv2.imread(prevFilePath)
        prev_img = cv2.resize(prev_img, (0, 0), fx=scale, fy=scale)
        prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_img_gray, img_gray, 0.5, 3, 15, 3, 5, 1.2, 0)


        # flowx, flowy = flow[..., 0], flow[..., 1]
        # flowx_crop, flowy_crop = flowx[y:y+128, x:x+64], flowy[y:y+128, x:x+64]
        #
        # hog_flow_x = hog(flowx_crop, orientations=9, pixels_per_cell=(8, 8),
        #                  cells_per_block=(2, 2), visualise=False)
        # hog_flow_y = hog(flowy_crop, orientations=9, pixels_per_cell=(8, 8),
        #                  cells_per_block=(2, 2), visualise=False)

        hsv = numpy.zeros_like(img)
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180/ numpy.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flowRGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        flow_gray = cv2.cvtColor(flowRGB, cv2.COLOR_BGR2GRAY)

        flow_gray_crop = flow_gray[y:y+128, x:x+64]

        hog_flow = hog(flow_gray_crop, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualise=False)

        desc = hog_gray + hog_flow

        set.append(desc)
    return set, [classType] * len(entries)


def get_hog_train_set(pos_filepath, neg_filepath, featureFile):
    p_features, p_classes = get_set(pos_filepath, 1)
    n_features, n_classes = get_set(neg_filepath, -1)
    print "positive samples", len(p_features)
    print "negative samples", len(n_features)

    features = p_features + n_features
    classes = p_classes + n_classes

    cPickle.dump((features, classes), open(featureFile, "wb"), protocol=2)

def trainSvmClassifier(feature_file, classifierFile):
    get_hog_train_set("hog_corpus_pos.txt", "hog_corpus_neg.txt", feature_file)

    obj = cPickle.load(open(feature_file, "rb"))

    features, classes = obj

    svc = svm.SVC(C=1.0, kernel='linear', probability=True).fit(features, classes)

    joblib.dump(svc, classifierFile)

trainSvmClassifier("hog_train_test_sum.dump", "hog_sum.dump")