import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import norm

import hyperparameters
import pedestrainDetection
from particleFilter import extract_bounding_box_image, ParticleFilter, compare_HSV_histograms
particle_num = hyperparameters.number_of_particles

def show_bb_on_image(frame,save):
    frameNum = '{0:04}'.format(frame)
    img = plt.imread(hyperparameters.video_frame_dir + '/frame_'+frameNum+'.jpg')
    bbInfo = pd.read_csv('tracker_out/frame_'+frameNum+'.txt', delimiter=' ', header=None)
    boundingBox = bbInfo.values
    if save == 0:
        pedestrainDetection.visualize_bounding_box(boundingBox, img, 1)
    else:
        img_out = pedestrainDetection.visualize_bounding_box(boundingBox, img, 0)
        plt.imsave(hyperparameters.bounded_image_out+'/frame_'+frameNum+'.jpg',img_out)

# boundingBox = pedestrainDetection.get_bounding_boxes_HOG(img, 1)


def calculate_similarity_score(particle, currframe, bounding_box):
    tracker_size = particle.bounding_box[2] * particle.bounding_box[3]
    detection_size = bounding_box[2] * bounding_box[3]
    tracker_detection_distance = norm.pdf(euclidean(bounding_box[:2], particle.bounding_box_position), 0,
                                          hyperparameters.position_var)
    size_diff = norm.pdf((tracker_size - detection_size) / tracker_size, 0, hyperparameters.position_var)
    gating = tracker_detection_distance * size_diff
    color_similarity = compare_HSV_histograms(extract_bounding_box_image(particle.frame, particle.bounding_box),
                                           extract_bounding_box_image(currframe, bounding_box))
    dist_similarity= 0
    for p in particle.particles:
        dist_similarity += norm.pdf(euclidean(p.pos,bounding_box[:2]), 0, hyperparameters.position_var)
    s = gating*(color_similarity+dist_similarity*hyperparameters.alpha)
    return s


def greedy_assignment(similarity_matrix):
    assignment = {}
    remain_detections = [i for i in range(similarity_matrix.shape[1])]
    remain_identites = [i for i in range(similarity_matrix.shape[0])]
    while len(remain_identites) != 0 and len(remain_identites) != 0 and np.sum(similarity_matrix)>0:
        max_id, max_det = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
        assignment[max_id] = max_det
        remain_identites.remove(max_id)
        remain_detections.remove(max_det)
        similarity_matrix[max_id, :] = 0
        similarity_matrix[:, max_det] = 0
    return assignment, remain_detections, remain_identites

def start_tracking():
    identities = []
    for frame in range(hyperparameters.number_of_frames):
        frame_id = '{0:04}'.format(frame)
        img = cv2.imread(hyperparameters.video_frame_dir + '/frame_' + frame_id + '.jpg')
        bounding_boxes = pedestrainDetection.get_bounding_boxes_HOG(img, 0)
        if len(identities) == 0:
            for i in range(len(bounding_boxes)):
                track = ParticleFilter(img, particle_num, i, frame, bounding_boxes[i, :])
                identities.append(track)
        else:
            similarity_matrix = np.zeros((len(identities), len(bounding_boxes)))
            for b in range(len(bounding_boxes)):
                for id in range(len(identities)):
                    similarity_matrix[id][b] = calculate_similarity_score(identities[id], img, bounding_boxes[b])
            assignment, remain_detections, remain_identites =greedy_assignment(similarity_matrix)
            for pair in assignment:
                identities[pair].update(img, bounding_boxes[assignment[pair]])
            if len(remain_detections)!=0:
                for det in remain_detections:
                    track = ParticleFilter(img,particle_num,len(identities),frame,bounding_boxes[det])
                    identities.append(track)
            if len(remain_identites)!=0:
                for id in remain_identites:
                    identities[id].detected = 0
        output = []
        for tr in identities:
            if tr.detected ==1:
                temp = np.zeros(6)
                temp[0] = frame
                temp[1:5] = tr.bounding_box
                temp[5] = tr.id
                output.append(temp)
        output = np.array(output)
        output = pd.DataFrame(output.astype('int'))
        output.to_csv(hyperparameters.output_path+'/'+'frame_'+frame_id+'.txt',sep=' ',header=False,index=False)
        show_bb_on_image(frame,1)

start_tracking()
# for i in range(200):
#     show_bb_on_image(i,1)
