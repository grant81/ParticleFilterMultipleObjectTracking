import cv2
import hyperparameters

image_folder = hyperparameters.bounded_image_out
video_name = 'traking_result_100_particles_high_performance_tracker.avi'

images = [image_folder+ '/frame_{0:04}.jpg'.format(i) for i in range(hyperparameters.number_of_frames) ]
frame = cv2.imread(images[0])
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter(video_name, fourcc, 15, (width,height))

for image in images:
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()