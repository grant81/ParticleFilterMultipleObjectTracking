import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import norm
import copy
import hyperparameters


def extract_bounding_box_image(img, bounding_box):
    ylim = int(bounding_box[1] + (bounding_box[3]))
    xlim = int(bounding_box[0] + (bounding_box[2]))
    if ylim > img.shape[0]:
        ylim = img.shape[0]
    if xlim > img.shape[1]:
        xlim = img.shape[1]
    out = img[int(bounding_box[1]):ylim, int(bounding_box[0]):xlim].copy()
    return out


def compare_HSV_histograms(bb_image1, bb_image2):
    new_bb_upper = bb_image1[:int(bb_image1.shape[0] / 2), :]
    new_bb_lower = bb_image1[int(bb_image1.shape[0] / 2):, :]
    hsv_upper = cv2.cvtColor(new_bb_upper, cv2.COLOR_BGR2HSV)
    hsv_lower = cv2.cvtColor(new_bb_lower, cv2.COLOR_BGR2HSV)
    hist_upper = cv2.calcHist([hsv_upper], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist_lower = cv2.calcHist([hsv_lower], [0, 1], None, [180, 256], [0, 180, 0, 256])
    curr_bb_upper = bb_image2[:int(bb_image2.shape[0] / 2), :]
    curr_bb_lower = bb_image2[int(bb_image2.shape[0] / 2):, :]
    hsv_upper_curr = cv2.cvtColor(curr_bb_upper, cv2.COLOR_BGR2HSV)
    hsv_lower_curr = cv2.cvtColor(curr_bb_lower, cv2.COLOR_BGR2HSV)
    hist_upper_curr = cv2.calcHist([hsv_upper_curr], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist_lower_curr = cv2.calcHist([hsv_lower_curr], [0, 1], None, [180, 256], [0, 180, 0, 256])
    s_score_lower = cv2.compareHist(hist_lower, hist_lower_curr, cv2.HISTCMP_BHATTACHARYYA)
    s_score_upper = cv2.compareHist(hist_upper, hist_upper_curr, cv2.HISTCMP_BHATTACHARYYA)
    a_score = np.exp(-20 * s_score_lower - 20 * s_score_upper)
    return a_score


class particle:
    def __init__(self, pos, speed):
        # self.weight = weight
        self.pos = pos
        self.speed = speed

    def set_pos(self, pos):
        if pos[0] < 0:
            pos[0] = 0
        if pos[1] < 0:
            pos[1] = 0
        self.pos = pos


class ParticleFilter:
    def __init__(self, frame, particle_num, detection_id, frame_num, bounding_box):
        self.frame = frame.copy()
        self.particle_num = particle_num
        self.id = detection_id
        self.frame_bun = frame_num
        self.bounding_box = bounding_box.copy()
        self.bounding_box_position = np.array([self.bounding_box[0], self.bounding_box[1]])
        self.bounding_box_size = np.array([self.bounding_box[2], self.bounding_box[3]])
        self.last_four_size = np.array([[self.bounding_box[2], self.bounding_box[3]] for k in range(4)])
        self.detected = 1
        weight = 1 / particle_num
        self.weights = np.ones(self.particle_num) * weight
        initial_pos = self.sample_init_position(self.bounding_box_position, particle_num)
        speed = self.init_motion()
        initial_speeds = self.sample_init_speed(speed, particle_num)
        self.particles = []
        for i in range(particle_num):
            pa = particle(pos=np.array([0, 0]), speed=np.array([initial_speeds[i][0], initial_speeds[i][1]]))
            pa.set_pos(initial_pos[i])
            self.particles.append(pa)

    def update(self, new_frame, bounding_box):
        self.proporgate()
        self.measure(new_frame, bounding_box)
        self.resample()
        bounding_box = self.get_bounding_box()
        self.detected = 1
        return bounding_box

    def proporgate(self):
        pos_noise = self.generate_noise(np.array([0, 0]), hyperparameters.position_var, self.particle_num)
        speed_noise = self.generate_noise(np.array([0, 0]), hyperparameters.speed_var, self.particle_num)
        for p in range(self.particle_num):
            self.particles[p].set_pos(self.particles[p].pos + self.particles[p].speed + pos_noise[p])
            self.particles[p].speed = self.particles[p].speed + speed_noise[p]

    def measure(self, new_frame, bounding_box):
        det_weight = np.zeros(self.particle_num)
        for p in range(self.particle_num):
            distance = euclidean(np.array([bounding_box[0], bounding_box[1]]), self.particles[p].pos)
            det_weight[p] = norm.pdf(distance, loc=0, scale=5) + 1e-100
        det_weight = det_weight / np.sum(det_weight)

        # appearance feature two part hsv histogram
        app_weight = np.zeros(self.particle_num)
        new_bb_image = extract_bounding_box_image(new_frame, bounding_box)
        new_bb_upper = new_bb_image[:int(new_bb_image.shape[0] / 2), :]
        new_bb_lower = new_bb_image[int(new_bb_image.shape[0] / 2):, :]
        hsv_upper = cv2.cvtColor(new_bb_upper, cv2.COLOR_BGR2HSV)
        hsv_lower = cv2.cvtColor(new_bb_lower, cv2.COLOR_BGR2HSV)
        hist_upper = cv2.calcHist([hsv_upper], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_lower = cv2.calcHist([hsv_lower], [0, 1], None, [180, 256], [0, 180, 0, 256])
        for p in range(self.particle_num):
            pos = self.particles[p].pos
            curr_size = self.get_avg_size()
            ylim = int(pos[1] + (curr_size[1]))
            xlim = int(pos[0] + (curr_size[0]))
            if ylim > self.frame.shape[0]:
                ylim = self.frame.shape[0]
            if xlim > self.frame.shape[1]:
                xlim = self.frame.shape[1]

            curr_bb_image = self.frame[int(pos[1]):ylim, int(pos[0]):xlim]
            curr_bb_upper = curr_bb_image[:int(curr_bb_image.shape[0] / 2), :]
            curr_bb_lower = curr_bb_image[int(curr_bb_image.shape[0] / 2):, :]
            hsv_upper_curr = cv2.cvtColor(curr_bb_upper, cv2.COLOR_BGR2HSV)
            hsv_lower_curr = cv2.cvtColor(curr_bb_lower, cv2.COLOR_BGR2HSV)
            hist_upper_curr = cv2.calcHist([hsv_upper_curr], [0, 1], None, [180, 256], [0, 180, 0, 256])
            hist_lower_curr = cv2.calcHist([hsv_lower_curr], [0, 1], None, [180, 256], [0, 180, 0, 256])
            s_score_lower = cv2.compareHist(hist_lower, hist_lower_curr, cv2.HISTCMP_BHATTACHARYYA)
            s_score_upper = cv2.compareHist(hist_upper, hist_upper_curr, cv2.HISTCMP_BHATTACHARYYA)
            a_score = np.exp(-20 * s_score_lower - 20 * s_score_upper)  # according to  A boosted particle filter paper
            app_weight[p] = a_score
        app_weight = app_weight / np.sum(app_weight)
        # update weights
        weights = np.add(hyperparameters.detection_weight_percent * det_weight,
                         hyperparameters.appearance_weight_percent * app_weight)
        weights = weights / np.sum(weights)
        self.weights = weights
        # for p in range(self.particle_num):
        #     self.particles[p].weight = weights[p]

    def resample(self):
        indexs = np.array([i for i in range(self.particle_num)])
        new_particle_list = []
        sampled_particle_index = np.random.choice(indexs, self.particle_num, p=self.weights)
        for i in sampled_particle_index:
            new_particle_list.append(copy.deepcopy(self.particles[i]))
        self.particles = copy.deepcopy(new_particle_list)
        self.weights = np.ones(self.particle_num)/self.particle_num

    def calculate_mean_pos(self):
        avg_position = np.zeros(2)
        for p in self.particles:
            avg_position += p.pos
        avg_position = avg_position / self.particle_num
        return avg_position

    def get_bounding_box(self):
        b_box = np.zeros(4)
        pos = self.calculate_mean_pos()
        size = self.get_avg_size()
        b_box[0] = pos[0]
        b_box[1] = pos[1]
        b_box[2] = size[0]
        b_box[3] = size[1]
        self.bounding_box_position = pos
        self.bounding_box_size = size
        self.bounding_box = b_box
        return b_box

    # TODO resample according to weight
    # TODO update bbox, frame, etc

    def init_motion(self, unitSpeed=1):
        speed = np.array([unitSpeed, unitSpeed])
        xdir, ydir, zdir = self.frame.shape
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

    def get_bounding_box_image(self):
        return extract_bounding_box_image(self.frame, self.bounding_box)
