import cv2
import numpy as np
import time


cap = cv2.VideoCapture(0)
prev_time = 0


bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)


def count_fingers(contour, drawing):
    
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is not None:
            
            finger_count = 0
            
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                
                
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                
                
                if angle <= np.pi / 2:
                    finger_count += 1
                    
                    cv2.circle(drawing, far, 5, [0, 0, 255], -1)
            
            
            return finger_count + 1
    return 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam")
        break
    
    
    frame = cv2.flip(frame, 1)
    
    
    drawing = frame.copy()
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
   
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
   
    roi_size = 300
    height, width = frame.shape[:2]
    roi_x = width // 2 - roi_size // 2
    roi_y = height // 2 - roi_size // 2
    
   
    cv2.rectangle(drawing, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 255, 0), 2)
    
   
    roi = thresh[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
    
    
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    finger_count = 0
    
    
    if contours:
        
        max_contour = max(contours, key=cv2.contourArea)
        
       
        if cv2.contourArea(max_contour) > 5000:
            
            offset_contour = max_contour.copy()
            for i in range(len(offset_contour)):
                offset_contour[i][0][0] += roi_x
                offset_contour[i][0][1] += roi_y
            
            
            cv2.drawContours(drawing, [offset_contour], 0, (0, 255, 0), 2)
            
            
            finger_count = count_fingers(offset_contour, drawing)
    
    
    cv2.putText(drawing, str(finger_count), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)
    
   
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(drawing, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
   
    cv2.imshow('Finger Counter', drawing)
    
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()