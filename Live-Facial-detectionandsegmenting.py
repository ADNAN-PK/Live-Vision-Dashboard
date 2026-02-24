import cv2
import numpy as np
import time
import os
import urllib.request

print("Checking for CNN Model files...")

# ==========================================
# 1. AUTO-DOWNLOAD CNN MODELS (For Python 3.14 compatibility) - Mohamed Adnan - CST4060 couse work - upgraded
# ==========================================
# Since we can't use MediaPipe, we use OpenCV's built-in Caffe DNN for Face Detection
prototxt_path = "deploy.prototxt"
caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(prototxt_path):
    print("Downloading deploy.prototxt...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", 
        prototxt_path
    )

if not os.path.exists(caffemodel_path):
    print("Downloading res10_300x300_ssd_iter_140000.caffemodel (~10MB)...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", 
        caffemodel_path
    )

# ==========================================
# 2. INITIALIZE MODELS & VISION PIPELINES
# ==========================================

# Model 1: Classic Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Model 2: Deep Neural Network (CNN) Face Detector
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Model 3: Background Subtractor (Motion-based Segmentation fallback)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=False)

# ==========================================
# 3. DEFINE 13 IMAGE FILTERS
# ==========================================

def apply_filter(img, filter_index):
    if filter_index == 0: # 1. Normal
        return img.copy()
    elif filter_index == 1: # 2. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filter_index == 2: # 3. Sobel Edge Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.magnitude(sobelx, sobely)
        sobel = np.uint8(np.absolute(sobel))
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    elif filter_index == 3: # 4. Canny Edge
        edges = cv2.Canny(img, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_index == 4: # 5. Gaussian Blur
        return cv2.GaussianBlur(img, (21, 21), 0)
    elif filter_index == 5: # 6. Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(img, -1, kernel)
    elif filter_index == 6: # 7. Sepia
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(img, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    elif filter_index == 7: # 8. Invert (Negative)
        return cv2.bitwise_not(img)
    elif filter_index == 8: # 9. Emboss
        kernel = np.array([[0,-1,-1], [1,0,-1], [1,1,0]])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        emboss = cv2.filter2D(gray, -1, kernel) + 128
        return cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)
    elif filter_index == 9: # 10. Pixelate
        h, w = img.shape[:2]
        temp = cv2.resize(img, (w // 15, h // 15), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    elif filter_index == 10: # 11. Sketch
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif filter_index == 11: # 12. Cartoon
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    elif filter_index == 12: # 13. Heatmap
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

filter_names = [
    "Normal", "Grayscale", "Sobel Edges", "Canny Edges", "Gaussian Blur", 
    "Sharpen", "Sepia", "Invert", "Emboss", "Pixelate", "Sketch", "Cartoon", "Heatmap"
]

# ==========================================
# 4. MAIN APPLICATION LOOP
# ==========================================

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_filter = 0
    pTime = 0

    print("\nStarting Live Vision Dashboard...")
    print("Controls:")
    print(" 'f' - Next Filter")
    print(" 'q' - Quit")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip horizontally for selfie-view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ------------------------------------------
        # PANEL 1: Classic OpenCV Haar Cascade
        # ------------------------------------------
        panel_1 = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, fw, fh) in faces:
            cv2.rectangle(panel_1, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(panel_1, "Haar Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(panel_1, "1. Haar Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ------------------------------------------
        # PANEL 2: OpenCV CNN (Deep Neural Network)
        # ------------------------------------------
        panel_2 = frame.copy()
        
        # Prepare image for DNN (Mean Subtraction & Scaling)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5: # 50% confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(panel_2, (startX, startY), (endX, endY), (255, 100, 0), 2)
                
                text = f"CNN: {confidence*100:.1f}%"
                cv2.putText(panel_2, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

        cv2.putText(panel_2, "2. CNN Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ------------------------------------------
        # PANEL 3: Motion Segmentation (Background Subtraction)
        # ------------------------------------------
        # Note: Since MediaPipe is unavailable, we use motion-based segmentation.
        # It detects moving foreground objects and replaces the static background.
        fg_mask = bg_subtractor.apply(frame)
        
        # Clean up the mask
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
        
        condition = fg_mask > 0
        bg_image = np.zeros(frame.shape, dtype=np.uint8)
        bg_image[:] = (40, 40, 80) # Dark blue virtual background
        
        # Expand condition to 3 channels to blend with frame
        panel_3 = np.where(condition[:, :, None], frame, bg_image)
        cv2.putText(panel_3, "3. Motion Segmenting", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 0), 2)

        # ------------------------------------------
        # PANEL 4: Filter Viewer
        # ------------------------------------------
        panel_4 = apply_filter(frame, current_filter)
        cv2.putText(panel_4, f"4. Filter: {filter_names[current_filter]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(panel_4, "Press 'f' to change", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

        # ==========================================
        # 5. COMBINE PANELS AND RENDER
        # ==========================================
        top_row = np.hstack((panel_1, panel_2))
        bottom_row = np.hstack((panel_3, panel_4))
        dashboard = np.vstack((top_row, bottom_row))

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        
        cv2.putText(dashboard, f"FPS: {int(fps)}", (20, dashboard.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Live Vision Dashboard (Python 3.14)", dashboard)

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            current_filter = (current_filter + 1) % len(filter_names)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()