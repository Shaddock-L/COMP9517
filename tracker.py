from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import matplotlib.pyplot as plt
from distance_detector import DistanceDetector
from _collections import deque
import numpy as np
from imutils.object_detection import non_max_suppression as nms


pts = [deque(maxlen=100) for _ in range(500)]
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def draw_bboxes(image, bboxes, face_bboxes, select_roi_points, select_roi_person, image_scale, line_thickness=None):
    # Plots one bounding box on image img
    line_thickness = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    font_thickness = max(line_thickness - 1, 1)  # font thickness
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]
    person_colors = {}
    global pts
    for bbox in bboxes:
        if bbox[-1] not in person_colors:
            person_colors[bbox[-1]] = color_lut[bbox[-1] % len(color_lut)]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), person_colors[bbox[-1]], thickness=line_thickness, lineType=cv2.LINE_AA)
        text_size = cv2.getTextSize('{} {}'.format(bbox[-2], bbox[-1]), 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + text_size[0], bbox[1] - text_size[1] - 3), person_colors[bbox[-1]], -1, cv2.LINE_AA)  # filled
        
        # trajectory
        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        #print(center)
        pts[bbox[-1]].append(center)
        for j in range(1, len(pts[bbox[-1]])):
            if pts[bbox[-1]][j-1] is None or pts[bbox[-1]][j] is None:
                print("no")
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(image, (pts[bbox[-1]][j-1]), (pts[bbox[-1]][j]), person_colors[bbox[-1]], thickness) 


        cv2.putText(image, '{} {}'.format(bbox[-2], bbox[-1]), (bbox[0], bbox[1] - 3), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)
    if len(select_roi_person) > 0:
        c1 = (int(select_roi_points[0][0]), int(select_roi_points[0][1]))
        c2 = (int(select_roi_points[1][0]), int(select_roi_points[1][1]))
        cv2.rectangle(image, c1, c2, (255, 0, 255), thickness=line_thickness-2, lineType=cv2.LINE_AA)
        cv2.putText(image, 'Select Region', (c1[0], c1[1]-5), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)
        cv2.putText(image, f'Total count:{len(face_bboxes)} current count:{len(bboxes)} ROI region count:{len(select_roi_person)}', (30, 40), 0, line_thickness / 3,
                [225, 255, 0], thickness=font_thickness, lineType=cv2.LINE_AA)
    else:
        cv2.putText(image, f'Total count:{len(face_bboxes)} current count:{len(bboxes)}', (30, 40), 0, line_thickness / 3,
                    [225, 255, 0], thickness=font_thickness, lineType=cv2.LINE_AA)

    


    return image

def draw_group_bboxes(image, groups, face_bboxes, centroids):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]
    person_colors = {}
    cnt = 0
    rm_dup = []

    #print(f'there are {len(groups)} groups')

    rects = []

    for group in groups:
        
        person_colors[tuple(group)] = color_lut[sum(tuple(group)) % len(color_lut)]
        #cnt += len(group)
        # draw large rectange to surround the group members
        top_left_x = 1e5
        top_left_y = -1e5
        bottom_right_x = -1e5
        bottom_right_y = 1e5
        for person in group:
            ind = face_bboxes.index(person)
            centroid = centroids[ind]
            x, y = centroid[0], centroid[1]
            top_left_x = min(top_left_x, x)
            top_left_y = max(top_left_y, y)
            bottom_right_x = max(bottom_right_x, x)
            bottom_right_y = min(bottom_right_y,y)
            if(ind not in rm_dup):
                rm_dup.append(ind)
                cnt += 1
            

        top_left_x = int(top_left_x)
        top_left_y  =int(top_left_y)
        bottom_right_x =int(bottom_right_x)
        bottom_right_y = int(bottom_right_y)

        top_left_x -= 50
        top_left_y += 80
        bottom_right_x += 50
        bottom_right_y -= 80
        rects.append((top_left_x,top_left_y,bottom_right_x,bottom_right_y))
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y),\
            person_colors[tuple(group)], thickness= 10 , lineType=cv2.LINE_AA)


    # cand = np.array([[x1, y1, x2, y2] for (x1, y1, x2, y2) in rects])
    # pick = nms(cand, probs=None, overlapThresh=0.2)
    # for (x1,y1,x2,y2) in pick:
    #     cv2.rectangle(image, (x1, y1), (x2, y2),\
    #         person_colors[tuple(group)], thickness= 10 , lineType=cv2.LINE_AA)


    return image, cnt

def draw_enter_leave(image, centroids):
    # draw a exclamation mark when a pedestrian enter or leave the scene
    img_row = image.shape[0]
    img_col = image.shape[1]
    min_distance = 80
    cnt = 0
    for center in centroids:
        x, y = int(center[0]), int(center[1])
        if (y < min_distance) or (y > img_row - min_distance) or (x < min_distance) or (x > img_col - min_distance):
            #print(center)
            left = 0  if  (y - 40) < 0 else (y - 40)
            right = img_col if (y + 40) > img_col else (y + 40)
            top = img_row if (x + 80) > img_row else (x + 40)
            bottom = x if (x + 40) > img_row else (x + 100)
            cnt += 1
            #print((left, right, top, bottom))
            for i in range(bottom, top):
               for j in range(left, right):
                   image[j][i][:] -= 100
            cv2.circle(image, (x, y - 30), 10, (255,255,255), -1)
            cv2.ellipse(image, (x , y-120), (50, 10), 90, 0, 360, (255, 255, 255), -1)
    return image




def update(target_detector, image, total_face_bboxes, face_bboxes, centroids, select_roi_points, image_scale, show_group_flag):

        _, bboxes = target_detector.detect(image)
        bbox_info = []
        new_faces = []
        confidences = []
        bboxes_info = []
        select_roi_person = []
        previous_face_bboxes = [item for item in face_bboxes]
        previous_centroids = [item for item in centroids]
        centroids = []
        if len(bboxes) > 0:
            for bbox in bboxes:
                x1,y1,x2,y2 = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2), bbox[2] - bbox[0], bbox[3] - bbox[1]
                
                if len(select_roi_points) > 0:
                    center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
                    if bbox[0] >= select_roi_points[0][0] and bbox[1] >= select_roi_points[0][1] and bbox[2] <= select_roi_points[1][0] and bbox[3] <= select_roi_points[1][1]:
                        #print((bbox[0],bbox[1],bbox[2],bbox[3]))
                    #if center[0] >= select_roi_points[0][0] and center[1] >= select_roi_points[0][1] and center[0] <= select_roi_points[1][0] and center[1] <= select_roi_points[1][1]:
                        select_roi_person.append(bbox)
                bbox_info.append([x1, y1, x2, y2])
                confidences.append(bbox[-1])

            bbox_info_tensor = torch.Tensor(bbox_info)
            confidences_tensor = torch.Tensor(confidences)

            # Pass detections to deepsort
            predictions = deepsort.update(bbox_info_tensor, confidences_tensor, image)

            for prediction in list(predictions):
                bboxes_info.append(
                    (*prediction[0:4], 'person', prediction[-1])
                )
                
                centroids.append((prediction[0] + (prediction[2] - prediction[0]) / 2, prediction[1] + (prediction[3] - prediction[1]) / 2))
                face_bboxes.append(prediction[-1])
                
                if prediction[-1] not in total_face_bboxes:
                    total_face_bboxes.append(prediction[-1])
                    new_faces.append(prediction)
            

                
        image = draw_bboxes(image, bboxes_info, total_face_bboxes, select_roi_points, select_roi_person, image_scale)

        # draw a exclamation mark when a pedestrian enter or leave the scene
        if len(centroids) > 0:
            image = draw_enter_leave(image, centroids)

        # draw rectangles which surround groups
        if len(previous_face_bboxes) > 0 and len(previous_centroids) > 0:
            dist = DistanceDetector(80, centroids, face_bboxes, previous_centroids, previous_face_bboxes)
            groups = dist.detect()
            if(show_group_flag == True):
                image, gourp_ped_count = draw_group_bboxes(image, groups, face_bboxes, centroids)
                ped_count = len(bbox_info)
                cv2.putText(image, f'Walking in groups:{gourp_ped_count} , walking alone: {ped_count - gourp_ped_count}', (50, 100), 0, 1,
                    [100, 255, 100], thickness=2, lineType=cv2.LINE_AA)
                

        return image, new_faces, face_bboxes, centroids
