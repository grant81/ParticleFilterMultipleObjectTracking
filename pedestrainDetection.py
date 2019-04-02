import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.object_detection import non_max_suppression

from hyperparameters import color


def get_bounding_boxes_HOG(image, verbose=0):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, padding=(8, 8), scale=1.05)
    # weights = np.reshape(weights,weights.shape[0])
    # bounding_box_dict = dict(zip(weights,rects))

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, overlapThresh=0.5)
    pick = np.array([[x, y, x1 - x, y1 - y] for (x, y, x1, y1) in pick])
    # pick = np.array([[int(x+w/2), int(y+h/2), w,h] for (x, y, w, h) in pick])

    if verbose != 0:
        # draw the final bounding boxs
        visualize_bounding_box(pick, orig, 1)
        # show some information on the number of bounding boxes
        print("[INFO]:{} original boxes, {} after suppression".format(len(rects), len(pick)))
    return pick


def visualize_bounding_box(bounding_boxses, frame, show=0):
    bounded_frame = frame.copy()
    b_boxes = bounding_boxses[:, 1:]

    for (x, y, w, h, id) in b_boxes:
        cv2.rectangle(bounded_frame, (x, y), (x + w, y + h), color[id % 7], 2)

    if show != 0:
        plt.imshow(bounded_frame)
        plt.show()
    return bounded_frame
