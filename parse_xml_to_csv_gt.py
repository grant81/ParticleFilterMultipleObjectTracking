import xml.etree.ElementTree as et

import numpy as np
import pandas as pd

import hyperparameters

xtree = et.parse(hyperparameters.xml_ground_truth_path_hard)
xroot = xtree.getroot()

for node in xroot:
    frame = int(node.attrib.get('number'))
    object_list = node.find('objectlist')
    ids_for_frame = []
    for object in object_list:
        curr_row = np.zeros(6)
        id = int(object.attrib.get('id'))
        bbox = object.find('box')
        h = float(bbox.attrib.get('h'))
        w = float(bbox.attrib.get('w'))
        xc = float(bbox.attrib.get('xc'))
        yc = float(bbox.attrib.get('yc'))
        curr_row[0] = frame
        curr_row[1] = int(xc - w / 2)
        curr_row[2] = int(yc - h / 2)
        curr_row[3] = int(w)
        curr_row[4] = int(h)
        curr_row[5] = int(id)
        ids_for_frame.append(curr_row)
    output = pd.DataFrame(np.array(ids_for_frame,np.int))
    frame_id = '{0:04}'.format(frame)
    output.to_csv(hyperparameters.csv_ground_truth_path + '/' + 'frame_' + frame_id + '.txt', sep=' ', header=False, index=False)
