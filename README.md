# Optical Flow & Hand Tracking Benchmark

This project explores **hand tracking in video** and **optical flow estimation** using both classical computer vision algorithms and deep learning models.  
Developed as part of the **Image, Audio and Video Processing (PIAV)** course, it benchmarks 5 approaches on the same data and provides a comparative analysis of **position** and **velocity**.


## 🎯 Objectives
- Track the **right hand** in the `people2` video using different algorithms.  
- Compare **position stability** and **velocity estimates** across methods.  
- Apply the same algorithms to the **FlyingChairs dataset** and evaluate against ground truth.


## 🧩 Methods implemented
- **YOLOv8-Pose** → detects people and tracks keypoint 10 (right hand).  
- **Lucas–Kanade (manual implementation)** → optical flow from first principles (Sobel gradients, normal equations).  
- **Lucas–Kanade (OpenCV)** → efficient sparse tracking with pyramids.  
- **Farnebäck** → dense optical flow, sampled at the keypoint.  
- **RAFT** → state-of-the-art deep learning optical flow model (torchvision pretrained).  

Velocity is computed as **Euclidean displacement × FPS**, allowing fair comparison across methods.


## 📊 Results at a glance
- **YOLOv8-Pose**: stable positions, higher and more dynamic velocity estimates.  
- **RAFT / Farnebäck / LK OpenCV**: consistent trajectories with moderate velocity fluctuations.  
- **LK Manual**: noticeable deviations, highlighting precision limitations of the hand-coded approach.  

On **FlyingChairs**, RAFT achieves results closest to the ground truth, with Farnebäck and LK producing reasonable but less precise flow fields.


## 🛠️ Tech stack
- **Python**  
- **PyTorch** + **torchvision (RAFT)**  
- **Ultralytics YOLOv8-Pose**  
- **OpenCV** (Lucas–Kanade, Farnebäck, video I/O)  
- **NumPy, Matplotlib**  
- **Google Colab** (GPU execution)


## 🚀 How to use
1. Clone this repository:
   ```bash
   git clone https://github.com/alebola/optical-flow-tracking-benchmark.git
   cd optical-flow-tracking-benchmark
   ```
2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python torch torchvision matplotlib numpy
   ```
3. Open and run the notebook:
   ```bash
   notebooks/optical_flow_tracking.ipynb
   ```
4. Required inputs:
   - videos/people2.mp4
   - A sample from the FlyingChairs dataset (00005_img1.ppm, 00005_img2.ppm, 00005_flow.flo)


## 📝 Notes
- This repository only includes a small sample of FlyingChairs (5 pairs). The full dataset (22,872 pairs) can be downloaded from the official source [FlyingChairs dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html). 
- This project was developed jointly with [David García Díaz](https://github.com/Davidgrcdz) as part of the **Image, Audio and Video Processing (PIAV)** course.


