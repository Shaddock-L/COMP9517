import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device
from tracker import update
import torchvision.transforms as T


class Detector:

    def __init__(self):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1
        self.weights = 'weights/yolov5m.pt'
        self.device = select_device('0' if torch.cuda.is_available() else 'cpu')
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        self.model = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def tracking(self, image, total_face_bboxes, face_bboxes, centroids, select_roi_points, image_scale, show_group_flag):
    
        results = {
            'face_bboxes': face_bboxes, 
            'centroids': centroids
        }

        image, faces, face_bboxes, centroids = update(self, image, total_face_bboxes, face_bboxes, centroids, select_roi_points, image_scale,show_group_flag)

        results['image'] = image
        results['faces'] = faces
        results['previous_face_bboxes'] = [item for item in results['face_bboxes']]
        results['previous_centroids'] = [item for item in results['centroids']]
        results['face_bboxes'] = face_bboxes
        results['total_face_bboxes'] = total_face_bboxes
        results['centroids'] = centroids
        
        return results

    def preprocess(self, image):

        image_orig = image.copy()
        image = letterbox(image, new_shape=self.img_size)[0]
        image = image[:, :, ::-1].transpose(2, 0, 1)
    
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.device)
        # transforms 
        # https://pytorch.org/vision/stable/transforms.html
        #image = T.ColorJitter(brightness=0.2, contrast=0.2)(image)
        #image = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(image)
        image = image.half() 
        image /= 255.0 
        if image.ndimension() == 3:
            image = image.unsqueeze(0)

        return image_orig, image

    def detect(self, image):

        image_orig, image = self.preprocess(image)

        predictions = self.model(image, augment=False)[0]
        predictions = non_max_suppression(predictions.float(), self.threshold, 0.4)

        results = []
        for prediction in predictions:

            if prediction is not None and len(prediction):
                prediction[:, :4] = scale_coords(
                    image.shape[2:], prediction[:, :4], image_orig.shape).round()

                for boxes in prediction:
                    classes_name = self.names[int(boxes[-1])]
                    if classes_name == 'person':
                        xyxy = [int(x) for x in boxes[0:4]]
                        results.append(
                            (*xyxy, classes_name, boxes[-2]))

        return image_orig, results

