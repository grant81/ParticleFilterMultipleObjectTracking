import pedestrainDetection
import matplotlib.pyplot as plt
import cv2
img = cv2.imread('example1.jpg')
boundingBox,weights = pedestrainDetection.get_bounding_boxes_HOG(img)
print(boundingBox)
print(weights)