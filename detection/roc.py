__author__ = 'mataevs'

import classifier
from classifier import *
import detection_checker
import utils
import tester_hog
import cascade_tester
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy
import optical_flow

def get_images(c, img_path, scales, subwindow=None):
    totalWindows = []
    for scale in scales:
        windows = c.getWindowsAndDescriptors(img_path, scale=scale, subwindow=subwindow)
        for i in range(0, len(windows)):
            # each window: (x,y,w,h), [featureDescriptor]
            x, y, w, h = windows[i][0]
            windows[i][0] = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
        totalWindows += windows
    return totalWindows

def getDecisionFunctionsForWindows(checkerFile, imageDirs, classifier, thresh, hog=True, subwindow=False):
    checker = detection_checker.Checker(checkerFile)
    fileLists = checker.getFileList()
    metadata = utils.parseMetadata(*imageDirs)

    trueClasses = []
    scores = numpy.array([])

    scales = [
        [0.45, 0.5, 0.55],
        [0.4, 0.45, 0.5],
        [0.3, 0.35],
        [0.3]
    ]
    scaleSteps = [35, 45, 65, 90]

    index = 0
    for imgPath in fileLists[:10]:
        print index
        index += 1
        tilt = int(metadata[imgPath]['tilt'])
        if tilt > 90:
            tilt = 90 - (tilt - 90)

        imgScales = []
        for i in range(0, len(scaleSteps)):
            if tilt < scaleSteps[i]:
                imgScales = scales[i]
                break

        boundingRect = None
        if subwindow:
            boundingRect = optical_flow.optical_flow(imgPath, utils.get_prev_img(imgPath))

        if hog:
            wf = tester_hog.getWindowsAndDescriptors(imgPath, imgScales, subwindow=boundingRect)
        else:
            wf = cascade_tester.get_images(classifier, imgPath, imgScales, subwindow=boundingRect)

        tc = checker.getWindowsClasses(imgPath, [w[0] for w in wf])

        df = computeClassifierDecisions(wf, classifier, thresh, hog=hog)

        print df

        trueClasses = trueClasses + tc
        scores = numpy.concatenate((scores, df))

    return trueClasses, scores

def computeClassifierDecisions(windows, classifier, thresh, hog=True):
    if hog:
        df = tester_hog.getDecisionFunction(classifier, windows)
    else:
        df = cascade_tester.get_decision_function(classifier, windows, thresh)
    return df

filepaths = [
        "/home/mataevs/captures/metadata/dump_05_05_01_50",
        "/home/mataevs/captures/metadata/dump_05_06_13_10",
        "/home/mataevs/captures/metadata/dump_10_06_11_47",
        "/home/mataevs/captures/metadata/dump_05_05_01_51",
        "/home/mataevs/captures/metadata/dump_05_06_13_15",
        "/home/mataevs/captures/metadata/dump_07_05_11_40",
        "/home/mataevs/captures/metadata/dump_10_06_11_48",
        "/home/mataevs/captures/metadata/dump_05_05_11_54",
        "/home/mataevs/captures/metadata/dump_05_06_13_20",
        "/home/mataevs/captures/metadata/dump_07_05_11_46",
        "/home/mataevs/captures/metadata/dump_10_06_12_16",
        "/home/mataevs/captures/metadata/dump_05_06_12_57",
        "/home/mataevs/captures/metadata/dump_05_06_13_21",
        "/home/mataevs/captures/metadata/dump_05_06_13_24",
        "/home/mataevs/captures/metadata/dump_16_06_14_57",
        "/home/mataevs/captures/metadata/dump_05_06_13_25",
        "/home/mataevs/captures/metadata/dump_07_05_12_03",
        "/home/mataevs/captures/metadata/dump_16_06_15_26",
        "/home/mataevs/captures/metadata/dump_05_06_13_28",
        "/home/mataevs/captures/metadata/dump_07_05_12_05"
    ]

def rocTest(checkerFile, imgDirs, classifierFile, hog=True, subwindow=False):
    if hog:
        c = tester_hog.load_classifier(classifierFile)
    else:
        c = classifier.loadCascadeClassifier(classifierFile)

    trueClasses, scores = getDecisionFunctionsForWindows(checkerFile, imgDirs, c, hog, subwindow=subwindow)

    print len(trueClasses), len(scores)

    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(trueClasses, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC HOG+Optical Flow SVM - Flow Subwindow')
    plt.legend(loc="lower right")
    plt.show()

def rocSoftCascadeTest(checkerFile, imgDirs, classifierFile, threshValues=[-1.0],  subwindow=False):
    c = classifier.loadSoftCascadeClassifier(classifierFile)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(0, len(threshValues)):
        thresh = threshValues[i]

        trueClasses, scores = getDecisionFunctionsForWindows(checkerFile, imgDirs, c, thresh, hog=False, subwindow=subwindow)

        print len(trueClasses), len(scores)

        # Compute micro-average ROC curve and ROC area
        fpr[i], tpr[i], _ = roc_curve(trueClasses, scores, pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(0, len(threshValues)):
        plt.plot(fpr[i], tpr[i], label='ROC curve for thresh=%0.2f (area = %0.2f)' % (threshValues[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC ICF Soft Cascade - 5000 features - 2000 weak classifiers - Variable Thresholds')
    plt.legend(loc="lower right")
    plt.show()

# rocTest("annotations.txt", filepaths, "cascade_classifier_100_500_2k.dump", False, subwindow=True)
# rocTest("annotations.txt", filepaths, "hog/hog_sum.dump", hog=True, subwindow=True)
rocSoftCascadeTest("annotations.txt", filepaths, "soft_cascade.dump", threshValues=[-1.0, -0.5, 0.0], subwindow=False)