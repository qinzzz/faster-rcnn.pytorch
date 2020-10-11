import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py


def add_detection_dict(objects_list, class_name, dets, feats,feature_lists,  thresh=0.1):
    count = 0
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        # feature = feats[i]
        score = dets[i, -1]
        if score >= thresh:
            count+=1
            # cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            # cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        # 1.0, (0, 0, 255), thickness=1)
            
            # output_format: class, score, x1,y1,x2,y2
            # line = {'class':class_name, 'score':str(score), 'bbox':str(bbox), 'feature':feature.tolist()}
            line = {'class':class_name, 'score':str(score), 'bbox':bbox}
            feature_lists.append(feats[i])
            # with open('test.json', 'w') as f:
            #     json.dump(line, f)
            # print(feature[:10])
            # input()
            # feature_list.append(feature)
            objects_list.append(line)
    return count

def gen_obj_dict(obj_detection):
    obj_detect_dict={}
    for img in obj_detection:
        img_id = int(img["image"].split('.')[0])
        # print(img_id)
        tmp={"bboxes":[], "classes":[], "scores":[]}
        for dic in img['objects']:
            bbox = [int(i) for i in dic["bbox"]]
            tmp["bboxes"].append(bbox)
            tmp["classes"].append(dic["class"])
            tmp["scores"].append(dic["score"])

        obj_detect_dict[img_id]=tmp
    return obj_detect_dict


def write_detections(image_list, imagename, objects_list):
    """write detections results to output file"""
    # outfile = open(filename, 'a')
    image_info={'image':imagename}
    image_info['objects']=objects_list
    print(len(objects_list))
    image_list.append(image_info)

    return image_list

