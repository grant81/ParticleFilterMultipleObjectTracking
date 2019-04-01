from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import matplotlib.pyplot as plt
import cv2
def get_bounding_boxes_HOG(image,verbose=0):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)
    weights = np.reshape(weights,weights.shape[0])
    # bounding_box_dict = dict(zip(weights,rects))

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=weights,overlapThresh=0.65)
    pick = np.array([[x,y,x1-x,y1-y] for (x,y,x1,y1) in pick])
    pick = np.array([[int(x+w/2), int(y+h/2), w,h] for (x, y, w, h) in pick])

    if verbose != 0 :
        # draw the final bounding boxs

        for (x, y, w, h) in pick:
            cv2.rectangle(orig, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 0, 255), 2)
        # show some information on the number of bounding boxes
        print("[INFO]:{} original boxes, {} after suppression".format(len(rects), len(pick)))
        # show the output images

        cv2.imshow("After NMS", orig)
        cv2.waitKey(0)

    return pick,weights