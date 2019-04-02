import os

import cv2


def split_video_by_frame(video, storage, frame_number_limit=1e99):
    # Playing video from file:
    cap = cv2.VideoCapture(video)
    # storage =

    try:
        if not os.path.exists(storage):
            os.makedirs(storage)
    except OSError:
        print('Error: Creating directory of data')

    currentFrame = 0
    while (currentFrame < frame_number_limit):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        # Saves image of the current frame in jpg file
        name = storage + str(currentFrame) + '.jpg'
        print('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# split_video_by_frame('./clips/2018-02-01-ana-ott-home_02.mp4', './data/frame/2018-02-01-ana-ott-home_02/')
