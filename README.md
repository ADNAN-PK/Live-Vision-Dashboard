<div align="center">
<h1>📷 Live Vision Dashboard (Python 3.14+)</h1>
<p><b>A dependency-light, 4-panel real-time computer vision dashboard built with pure OpenCV.</b></p>

</div>

📖 About The Project

This project creates a high-performance, 4-panel computer vision dashboard using your webcam. It was specifically engineered for Python 3.14 compatibility, bypassing heavy machine learning dependencies (like dlib or mediapipe which often lack pre-compiled binaries for bleeding-edge Python versions).

Instead, it relies purely on OpenCV and NumPy, automatically downloading the necessary lightweight Caffe models on its first run to provide Deep Neural Network face tracking and motion segmentation.

The 4-Panel Grid

Top-Left (Haar Tracking): Uses classic OpenCV Haar Cascades to find faces and draw bounding boxes.

Top-Right (CNN Tracking): Uses OpenCV's built-in Deep Neural Network (DNN) module with a pre-trained Caffe model for highly accurate face detection with confidence scores.

Bottom-Left (Motion Segmentation): Uses MOG2 Background Subtraction to detect moving foreground objects and dynamically replace static backgrounds with a studio blue backdrop.

Bottom-Right (Interactive Filters): A dynamic viewport displaying 13 unique, mathematically applied image filters.

✨ Key Features

Zero Dependency Hell: Requires only opencv-python and numpy. No CMake, C++ build tools, or complex virtual environments needed.

Auto-Downloading Models: Automatically fetches the required .prototxt and .caffemodel files (~10MB) directly from OpenCV's official GitHub on the first run.

Real-time Performance: Optimized with NumPy array operations to maintain smooth FPS while rendering a 2x2 multi-processing grid.

🚀 Getting Started

1. Prerequisites

You only need two standard Python libraries installed on your system.

pip install opencv-python numpy


2. Running the Application

Ensure your webcam is connected and not currently being used by another application, then run the script:

python live_vision_314.py


(On its first run, it will pause for a few seconds to download the DNN face detection models before opening your webcam).

🎮 Controls

Once the dashboard window opens, ensure the window is selected/focused, and use your keyboard to control the application:

f - Cycle through the 13 available live image filters on the bottom-right panel.

q - Quit and safely close the webcam application.

🎨 Supported Live Filters

Pressing f cycles through the following real-time mathematical operations applied via OpenCV and NumPy:

Normal (Pass-through)

Grayscale (cv2.cvtColor)

Sobel Edges (Calculates horizontal/vertical gradients)

Canny Edges (Classic edge detection algorithm)

Gaussian Blur (Smoothes pixels using a Gaussian kernel)

Sharpen (Enhances edges via a custom 2D convolution matrix)

Sepia (Applies a vintage color transformation matrix)

Invert (Creates a photo negative)

Emboss (Creates a 3D shadow/highlight effect)

Pixelate (Downscales and upscales using nearest-neighbor interpolation)

Sketch (Creates a pencil drawing effect by dividing grayscale and inverted blurs)

Cartoon (Combines bilateral filtering for color smoothing with adaptive thresholding for ink edges)

Heatmap (Applies a pseudocolor jet map based on pixel intensity)

<div align="center">
<i>Built with ❤️ using OpenCV and Python.</i>
</div>
