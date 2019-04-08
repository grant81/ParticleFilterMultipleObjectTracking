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


def show_bb_on_image(frame, tracker_out_dir, video_frame_dir, save):
    img = plt.imread(video_frame_dir + '/frame_' + frame + '.jpg')
    bbInfo = pd.read_csv(tracker_out_dir + '/frame_' + frame + '.txt', delimiter=' ', header=None)
    boundingBox = bbInfo.values
    if save == 0:
        pedestrainDetection.visualize_bounding_box(boundingBox, img, 1)
    else:
        img_out = pedestrainDetection.visualize_bounding_box(boundingBox, img, 0)
        plt.imsave(hyperparameters.bounded_image_out + '/frame_' + frame + '.jpg', img_out)


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
    dist_similarity = 0
    for p in particle.particles:
        dist_similarity += norm.pdf(euclidean(p.pos, bounding_box[:2]), 0, hyperparameters.position_var)
    s = gating * (color_similarity + dist_similarity * hyperparameters.alpha)
    return s


def greedy_assignment(similarity_matrix):
    assignment = {}
    remain_detections = [i for i in range(similarity_matrix.shape[1])]
    remain_identites = [i for i in range(similarity_matrix.shape[0])]
    while len(remain_identites) != 0 and len(remain_identites) != 0 and np.sum(similarity_matrix) > 0:
        max_id, max_det = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
        assignment[max_id] = max_det
        remain_identites.remove(max_id)
        remain_detections.remove(max_det)
        similarity_matrix[max_id, :] = 0
        similarity_matrix[:, max_det] = 0
    return assignment, remain_detections, remain_identites


def start_tracking(video_frame_dir, start_frame=0, output_dir=hyperparameters.tracker_output_path, tracker=0):
    identities = []
    for frame in range(hyperparameters.number_of_frames):
        f_id = '{0:04}'.format(frame + start_frame)
        det_id = '{0:04}'.format(frame)
        img = cv2.imread(video_frame_dir + '/frame_' + f_id + '.jpg')
        if tracker == 0:
            bounding_boxes = pedestrainDetection.get_bounding_boxes_HOG(img, 0)
        else:
            bounding_boxes = pd.read_csv(hyperparameters.high_performance_detetion_path + '/frame_' + det_id + '.txt',
                                         delimiter=' ', header=None)
            bounding_boxes = bounding_boxes.values
        if len(identities) == 0:
            for i in range(len(bounding_boxes)):
                track = ParticleFilter(img, particle_num, i, frame, bounding_boxes[i, :])
                identities.append(track)
        else:
            similarity_matrix = np.zeros((len(identities), len(bounding_boxes)))
            for b in range(len(bounding_boxes)):
                for id in range(len(identities)):
                    if identities[id].removed == 0:
                        similarity_matrix[id][b] = calculate_similarity_score(identities[id], img, bounding_boxes[b])
                    else:
                        similarity_matrix[id][b] = 0
            assignment, remain_detections, remain_identites = greedy_assignment(similarity_matrix)
            for pair in assignment:
                identities[pair].update(img, bounding_boxes[assignment[pair]])
            if len(remain_detections) != 0:
                for det in remain_detections:
                    # making sure new identities are assigned to new detection appears at the boundaries
                    curr_det = bounding_boxes[det]
                    if img.shape[1] - curr_det[0] < hyperparameters.initial_detection_boundary_distance or curr_det[
                        0] < hyperparameters.initial_detection_boundary_distance or img.shape[0] - curr_det[
                        1] < hyperparameters.initial_detection_boundary_distance or \
                            curr_det[1] < hyperparameters.initial_detection_boundary_distance:

                        track = ParticleFilter(img, particle_num, len(identities), frame, bounding_boxes[det])
                        identities.append(track)

            if len(remain_identites) != 0:
                for id in remain_identites:
                    identities[id].detected -= 1
                    if -1 * identities[id].detected >= hyperparameters.untracked_id_life_cycle:
                        identities[id].removed = 1
        output = []
        for tr in identities:
            if tr.detected == 1:
                temp = np.zeros(6)
                temp[0] = frame
                temp[1:5] = tr.bounding_box
                temp[5] = tr.id
                output.append(temp)
        output = np.array(output)
        output = pd.DataFrame(output.astype('int'))
        output.to_csv(output_dir + '/' + 'frame_' + f_id + '.txt', sep=' ', header=False, index=False)
        show_bb_on_image(f_id, output_dir, video_frame_dir, 1)
        print('frame ' + f_id + ' finish tracking')


# show_bb_on_image(0,0)
start_tracking(video_frame_dir=hyperparameters.video_frame_dir, output_dir=hyperparameters.tracker_output_path, start_frame=0,
               tracker=1)
# for i in range(200):
#     show_bb_on_image(i,1)
