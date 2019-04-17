num_of_state = 4
position_var = 7.0
speed_var = 2.0
detection_weight_percent = 0.6
appearance_weight_percent = 0.4
number_of_frames = 769
number_of_particles = 100
alpha =1
image_size = [576,768]
# image_size = [640,480]
untracked_id_life_cycle = 4
color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(0,0,0),(255,255,255),(100,200,0),(200,100,0),(100,0,200)]
#making sure new identities are assigned to new detection appears at the boundaries
initial_detection_boundary_distance = 100
# output_path = 'tracker_out'
tracker_output_path = 'tracker_output/tracker_out'
bounded_image_out = 'output_frames/easy'
video_frame_dir= 'frames/pets_frames'
high_performance_detetion_path = 'high_performance_detection/faster_rcnn_nas_lowproposals_coco_out_PETS'
xml_ground_truth_path = 'xml_gt/PETS2009-S2L1.xml'
xml_ground_truth_path_hard = 'xml_gt/PETS2009-S2L1.xml'
csv_ground_truth_path ='ground_truth/PETS_gt'


# dataset PETS2009 http://cs.binghamton.edu/~mrldata/public/PETS2009/
# dataset TUD https://www.d2.mpi-inf.mpg.de/node/428
# ground truth http://www.milanton.de/data/