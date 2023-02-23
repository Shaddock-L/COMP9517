from scipy.spatial import distance as dist
import numpy as np
import cv2
from utils.general import non_max_suppression


class DistanceDetector:
    def __init__(self, min_distance, centroids, face_bboxes, previous_centroids, previous_face_bboxes):
        self.min_distance = min_distance
        self.centroids = centroids
        self.previous_centroids = previous_centroids
        self.face_bboxes = face_bboxes
        self.previous_face_bboxes = previous_face_bboxes
        
    def calculate_group(self, centroids, face_bboxes):
        distances = dist.cdist(np.array(centroids), np.array(centroids), metric='euclidean')
        groups = set()
        add_flag = True
        if len(centroids) >= 2:
            for i in range(len(centroids)):
                g = []
                g.append(face_bboxes[i])
                for j in range(len(centroids)):
                    if i != j and distances[i, j] < self.min_distance:
                        for elem in groups:
                            if face_bboxes[j] in elem:
                                add_flag = False
                        if add_flag == True:
                            g.append(face_bboxes[j])
                        else:
                            add_flag = True
                            
                if len(g) > 1:
                    # avoid dup
                    temp = set(g)
                    g = list(temp)
                    g = sorted(g, key=lambda x: x)
                    groups.add(tuple(g))
        return groups
    
    def detect(self):
        groups = self.calculate_group(self.centroids, self.face_bboxes)
        previous_groups = self.calculate_group(self.previous_centroids, self.previous_face_bboxes)
        # stay close over 1 frame -> group
        results = set()
        for group in groups:
            if group in previous_groups:
                results.add(group)
        i = 0
        j = 0
        results = list(results)
        while i < len(results):
            while j < len(results):
                if i != j and (set(list(results[i])) & set(list(results[j])) != set()):
                    # combine together
                    temp = set(results[i]) | set(results[j])
                    results.remove(results[j])
                    results.append(temp)
                j += 1
            i += 1
        
        return results
                
            
                
            
        
        
    