import hyperparameters
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.spatial.distance import euclidean
import cv2

class particle:
    def __init__(self, pos, speed, weight=0):
        self.weight = weight
        self.pos = pos
        self.speed = speed


class ParticleFilter:
    def __init__(self, frame, particle_num, detection_id, frame_num, bounding_box):
        self.frame = frame.copy()
        self.particle_num = particle_num
        self.detection_id = detection_id
        self.frame_bun = frame_num
        self.bounding_box = bounding_box.copy()

        self.bounding_box_position = np.array([self.bounding_box[0], self.bounding_box[1]])
        self.size = np.array([self.bounding_box[2], self.bounding_box[3]])
        self.last_four_size = np.array([[self.bounding_box[2], self.bounding_box[3]] for k in range(4)])
        weight = 1 / particle_num
        initial_pos = self.sample_init_position(self.bounding_box_position, particle_num)
        speed = self.init_motion()
        initial_speeds = self.sample_init_speed(speed, particle_num)
        self.particles = [
            particle(pos=np.array([initial_pos[i][0], initial_pos[i][1]]),
                     speed=np.array([initial_speeds[i][0], initial_speeds[i][1]]),
                     weight=weight) for i in range(particle_num)]

    def proporgate(self):
        pos_noise = self.generate_noise(np.array([0, 0]), hyperparameters.position_var, self.particle_num)
        speed_noise = self.generate_noise(np.array([0, 0]), hyperparameters.speed_var, self.particle_num)
        for p in range(self.particle_num):
            self.particles[p].pos = self.particles[p].pos + self.particles[p].speed + pos_noise[p]
            self.particles[p].speed = self.particles[p].speed + speed_noise[p]

    def measure(self, new_frame, frame, bounding_box):
        det_weight = np.zeros(self.particle_num)
        for p in range(self.particle_num):
            distance = euclidean(np.array([bounding_box[0], bounding_box[1]]), self.particles[p].pos)
            det_weight[p] = norm.pdf(distance, loc=0, scale=5) + 1e-100
        det_weight = det_weight / np.sum(det_weight)

        #appearance feature two part hsv histogram
        app_weight = np.zeros(self.particle_num)
        new_bb_image = new_frame[
                       int(bounding_box[1] - bounding_box[3] / 2):int(bounding_box[1] + (bounding_box[3] / 2)),
                       int(bounding_box[0] - bounding_box[2] / 2):int(bounding_box[0] + (bounding_box[2] / 2))]
        new_bb_upper = new_bb_image[:int(new_bb_image.shape[0]/2),:]
        new_bb_lower = new_bb_image[int(new_bb_image.shape[0]/2):,:]
        hsv_upper = cv2.cvtColor(new_bb_upper, cv2.COLOR_BGR2HSV)
        hsv_lower = cv2.cvtColor(new_bb_lower, cv2.COLOR_BGR2HSV)
        hist_upper = cv2.calcHist([hsv_upper], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_lower = cv2.calcHist([hsv_lower], [0, 1], None, [180, 256], [0, 180, 0, 256])
        for p in range(self.particle_num):
            pos = self.particles[p].pos
            curr_size = self.get_avg_size()
            curr_bb_image = self.frame[int(pos[1] - curr_size[1] / 2):int(pos[1] + (curr_size[1] / 2)),
                            int(pos[0] - curr_size[0] / 2):int(pos[0] + (curr_size[0] / 2))]
            curr_bb_upper = curr_bb_image[:int(curr_bb_image.shape[0] / 2), :]
            curr_bb_lower = curr_bb_image[int(curr_bb_image.shape[0] / 2):, :]
            hsv_upper_curr = cv2.cvtColor(curr_bb_upper, cv2.COLOR_BGR2HSV)
            hsv_lower_curr = cv2.cvtColor(curr_bb_lower, cv2.COLOR_BGR2HSV)
            hist_upper_curr = cv2.calcHist([hsv_upper_curr], [0, 1], None, [180, 256], [0, 180, 0, 256])
            hist_lower_curr = cv2.calcHist([hsv_lower_curr], [0, 1], None, [180, 256], [0, 180, 0, 256])
            s_score_lower = cv2.compareHist(hist_lower, hist_lower_curr, cv2.CV_COMP_BHATTACHARYYA)
            s_score_upper = cv2.compareHist(hist_upper, hist_upper_curr, cv2.CV_COMP_BHATTACHARYYA)
            a_score = np.exp(-20*s_score_lower-20*s_score_upper)#according to  A boosted particle filter paper
            app_weight[p] = a_score
        app_weight = app_weight/np.sum(app_weight)
        #update weights
        weights = np.add(hyperparameters.detection_weight_percent*det_weight,hyperparameters.appearance_weight_percent*app_weight)
        for p in range(self.particle_num):
            self.particles[p].weight = weights[p]


    def resample(self):
        pass
    #TODO resample according to weight
    #TODO update bbox, frame, etc


    def init_motion(self, unitSpeed=1):
        speed = np.array([unitSpeed, unitSpeed])
        [xdir, ydir] = self.frame.shape
        if self.bounding_box[0] < xdir:
            speed[0] = -1 * unitSpeed
        if self.bounding_box[1] > ydir:
            speed[1] = -1 * unitSpeed
        return speed

    def sample_init_position(self, mean, size=None):
        pos = np.random.multivariate_normal(mean=mean,
                                            cov=np.identity(2) * hyperparameters.position_var,
                                            size=size)
        return pos

    def sample_init_speed(self, mean, size=None):
        speed = np.random.multivariate_normal(mean=mean,
                                              cov=np.identity(2) * hyperparameters.speed_var,
                                              size=size)
        return speed

    def generate_noise(self, mean, var, size=None):
        noise = np.random.multivariate_normal(mean=mean, cov=np.identity(2) * var,
                                              size=size)
        return noise

    def get_avg_size(self):
        return np.average(self.last_four_size, axis=0)
