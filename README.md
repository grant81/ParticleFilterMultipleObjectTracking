This is an implementation of the paper Robust Tracking-by-Detection using a Detector Confidence Particle Filter by Breitenstein et. al.

Project dependency: OpenCV
1. To run object tracking, run main.py
2. to evaluate the tracker performance run evalucation.py
3. to convert frames to video run frames_to_video.py
4. to parse the xml ground truth data to the csv format which this project uses, run parse_xml_to_csv.py
5. to change hyparameters or input/output folder path, change hyperparameters.py
6. to split a video into frames run split_video_by_frame

data format

frame_0000.txt

0 499 157 31 75 9

0 258 218 32 88 15

0 633 241 42 81 19

the columns are

frame number, x position of the left corner of the bounding box, y position of the bounding box, x center position, y center position, identity
