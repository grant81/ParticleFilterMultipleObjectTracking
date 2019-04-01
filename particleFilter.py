import hyperparameters
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm


class particle:
    def __init__(self, weight=0, state=np.zeros(hyperparameters.num_of_state)):
        self.weight = weight
        self.state = state


class ParticleFilter:
    def __init__(self, frame, particle_num, detection_id, frame_num, bounding_box):
        self.frame = frame
        self.particle_num = particle_num
        weight = 1 / particle_num
        initial_pos = np.random.multivariate_normal(mean=[bounding_box[0], bounding_box[1]],
                                                    cov=np.identity(2) * hyperparameters.position_var,
                                                    size=particle_num)
        speed = self.init_motion()
        self.particles = [
            particle(state=np.array([initial_pos[i][0], initial_pos[i][1], speed[0], speed[1]]), weight=weight) for i in
            range(particle_num)]
        self.detection_id = detection_id
        self.frame_bun = frame_num
        self.bounding_box = bounding_box
        self.bounding_box_img = np.zeros((10))

    def proporgate(self):
        pass

    def init_motion(self, unitSpeed=1):
        speed = np.array([unitSpeed, unitSpeed])
        [xdir, ydir] = self.frame.shape
        if self.bounding_box[0] < xdir:
            speed[0] = -1 * unitSpeed
        if self.bounding_box[1] > ydir:
            speed[1] = -1 * unitSpeed
        return speed
