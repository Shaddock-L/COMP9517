from Detector import Detector
import imutils
import cv2
import os
import keyboard

print_flag_q = True
show_group_flag = True

def main():
    name = 'PedestrianDetector'
    detector = Detector()
    
    face_bboxes = []
    centroids = []
    
    global print_flag_q
    global show_group_flag
    position1 = None
    position2 = None
    videoWriter = None
    select_roi_points = []
    total_face_bboxes = []
    print('[INFO] starting video stream...')
    for i,file in enumerate(os.listdir('step_images/test/STEP-ICCV21-01')):
        image = cv2.imread(os.path.join('step_images/test/STEP-ICCV21-01', file))
        
        image_size = image.shape[:2]
        image_scale = (image_size[0] / 640, image_size[1] / 640)
        if position1 == None and position2 == None and cv2.waitKey(30)>=0:
            # press space to start selecting roi, press space after selecting the roi, press q to reset roi
            r = cv2.selectROI(image,showCrosshair = False) 
            position1 = (int(r[0]), int(r[1]))
            position2 = (int(r[0]+r[2]), int(r[1]+r[3]))
            select_roi_points.append(position1)
            select_roi_points.append(position2)
            
            # tell the user pressing q to reset the roi
            if print_flag_q == True:
                print('Press q to reselect the roi')
                print_flag_q = False

        # reset the roi info
        if keyboard.is_pressed('q'):
            select_roi_points = []
            position1 = None
            position2 = None
            print_flag_q = True

        # showing group rects / or not
        if keyboard.is_pressed('e'):
            show_group_flag = not show_group_flag
        
        result = detector.tracking(image, total_face_bboxes, face_bboxes, centroids, select_roi_points, image_scale, show_group_flag)
        face_bboxes = result['face_bboxes']
        total_face_bboxes = result['total_face_bboxes']
        centroids = result['centroids']
        image = result['image']
        # save last 20% images to calculate the metrics
        # if i > 400:
        #     print(f"start to store img{i}")
        #     cv2.imwrite(f"./results_ICCV21-07_p1/result{i}.jpg",image)

        # image = imutils.resize(image, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, 30, (image.shape[1], image.shape[0]))

        videoWriter.write(image)
        cv2.imshow(name, image)
        cv2.waitKey(30)
        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            break

    videoWriter.release()
    cv2.destroyAllWindows()
    print('[INFO] video writer released')
if __name__ == '__main__':
    
    main()