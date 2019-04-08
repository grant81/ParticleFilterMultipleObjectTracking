import numpy as np
from hyperparameters import *
import pandas as pd


# Run IOU on every tracked object and ground truth object at each frame, evaluate greedily,
# if identity switched register a id switch change object id
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = np.array([boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]])
    boxB = np.array([boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]])
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


class matches:
    def __init__(self, gt, tr, confidence):
        self.gt = gt
        self.tr = tr
        self.confidence = confidence


def greedy_assignment(similarity_matrix):
    assignment = []
    remaining_tr_id = [i for i in range(similarity_matrix.shape[1])]
    remaining_gt_id = [i for i in range(similarity_matrix.shape[0])]
    while len(remaining_gt_id) != 0 and len(remaining_gt_id) != 0 and np.sum(similarity_matrix) > 0:
        max_id, max_det = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
        new_match = matches(max_id, max_det, similarity_matrix[max_id, max_det])
        assignment.append(new_match)
        remaining_gt_id.remove(max_id)
        remaining_tr_id.remove(max_det)
        similarity_matrix[max_id, :] = 0
        similarity_matrix[:, max_det] = 0
    return assignment, remaining_tr_id, remaining_gt_id


object_gt_correspondance = {}
precision = 0
totalFP = 0
totalFN = 0
totalIDSW = 0
totalTP = 0
MOTA = 0
total_detections = 0
for frame in range(0, number_of_frames):
    FP = 0
    FN = 0
    TP = 0
    IDSW = 0
    f_id = '{0:04}'.format(frame)
    tracker_frame = pd.read_csv(tracker_output_path + '/frame_' + f_id + '.txt', delimiter=' ', header=None)
    ground_truth = pd.read_csv(csv_ground_truth_path + '/frame_' + f_id + '.txt', delimiter=' ', header=None)
    tracker_data = tracker_frame.values
    ground_truth_data = ground_truth.values
    tracker_data = tracker_data[:, 1:]
    ground_truth_data = ground_truth_data[:, 1:]
    IOU_score_matrix = np.zeros((ground_truth_data.shape[0], tracker_data.shape[0]))
    for i in range(ground_truth_data.shape[0]):
        for j in range(tracker_data.shape[0]):
            IOU_score_matrix[i][j] = bb_intersection_over_union(ground_truth_data[i, :4], tracker_data[j, :4])
    if frame == 0:
        assignment, remaining_tr_id, remaining_gt_id = greedy_assignment(IOU_score_matrix)
        FP += len(remaining_tr_id)
        FN += len(remaining_gt_id)
        for a in assignment:
            if a.confidence < 0.5:
                FN += 1
            else:
                object_gt_correspondance[ground_truth_data[a.gt, 4]] = {tracker_data[a.tr, 4]}
                TP += 1
    else:
        assignment, remaining_tr_id, remaining_gt_id = greedy_assignment(IOU_score_matrix)
        FP += len(remaining_tr_id)
        FN += len(remaining_gt_id)
        for a in assignment:
            if a.confidence < 0.5:
                FN += 1
            else:
                gt_id = ground_truth_data[a.gt, 4]
                tr_id = tracker_data[a.tr, 4]
                if gt_id in object_gt_correspondance:
                    if tr_id not in object_gt_correspondance[gt_id]:
                        object_gt_correspondance[gt_id].add(tr_id)
                        IDSW += 1
                    else:
                        TP += 1
                else:
                    object_gt_correspondance[gt_id] = {tr_id}
                    TP += 1
    total_detections += ground_truth_data.shape[0]
    MOTA = (MOTA * frame + (FP + FN + IDSW) / ground_truth_data.shape[0]) / (frame + 1)
    totalFN += FN
    totalFP += FP
    totalIDSW += IDSW
    totalTP += TP
    # precision = (precision * frame + TP / (ground_truth_data.shape[0])) / (frame + 1)

MOTA = 1 - MOTA
print('MOTA: {}, Precision: {}, FN: {}, FP: {}, IDSW: {}'.format(MOTA, totalTP / (totalTP + totalFP),
                                                                 totalFN / total_detections,
                                                                 totalFP / total_detections, totalIDSW))
