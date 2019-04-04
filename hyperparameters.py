num_of_state = 4
position_var = 7.0
speed_var = 2.0
detection_weight_percent = 0.6
appearance_weight_percent = 0.4
number_of_frames = 700
number_of_particles = 100
alpha =1
image_size = [576,768]
# image_size = [640,480]
untracked_id_life_cycle = 4
color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(0,0,0),(255,255,255),(100,200,0),(200,100,0),(100,0,200)]
output_path = 'tracker_out'
output_path_TUD = 'tracker_out_TUD'
bounded_image_out = 'bounded_image_out'
video_frame_dir = 'pets_frames'
video_frame_dir_TUD = 'TUD_frames'
tracking_Out = 'tracking_Out'
high_performance_detetion_path = 'faster_rcnn_nas_lowproposals_coco_out'
# high_performance_detetion_path = 'faster_rcnn_nas_lowproposals_coco_out_TUD'
xml_ground_truth_path = 'xml_gt/PETS2009-S2L1.xml'
xml_ground_truth_path_hard = 'xml_gt/PETS2009-S2L2.xml'
csv_ground_truth_path ='PETS_gt_hard'


# dataset PETS2009 http://cs.binghamton.edu/~mrldata/public/PETS2009/
# dataset TUD https://www.d2.mpi-inf.mpg.de/node/428
# ground truth http://www.milanton.de/data/