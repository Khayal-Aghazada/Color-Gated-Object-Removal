# Object Removal CV

<img width="346" height="346" alt="image" src="https://github.com/user-attachments/assets/4c0cdc73-b1a1-4d29-a6af-b56f170eedef" />
<img width="346" height="346" alt="image" src="https://github.com/user-attachments/assets/6ab7cfaa-902c-4889-a2fe-ad36ee5deeac" />


<p align="center">(Add a screenshot or GIF of your object removal demo in action here. Place your image at <code>assets/images/object_remove_demo.jpg</code>.)</p>

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)

## 🧼 About the Project

** Object Removal CV** is a Python-based computer vision application that removes moving or colored foreground objects from video streams, revealing a clean static background. It is designed for video editing, surveillance, background modeling, and educational demonstrations.

The tool leverages clean plate extraction, color gating (e.g., purple cloth detection), and foreground differencing to identify and remove unwanted objects, providing immediate visual feedback from both video files and live camera feeds.

---

## 🎯 Features

- 🧹 Real-time object removal from video
- 🟣 Supports cloth-based removal (e.g., purple invisibility effect)
- 📹 Works with webcam or video file input
- 🎨 Color gating and frame differencing-based detection
- 🖼️ Multiple view modes: `output`, `mask`, `triptych`, `original`
- 🧩 Modular codebase for easy extension
- 📦 Example videos and screenshots included

---

## 🧠 Use Cases

- Invisibility cloak effect with colored fabric
- Background subtraction and surveillance
- Clean plate creation and modeling
- CV/AI educational projects and tutorials
- Motion filtering in static scenes

---

## 🟣 Color-Gated Object Removal

This demo can **remove objects covered with a specific color**, such as **a purple cloth**. When a person or item is covered with purple fabric, the system detects those regions via HSV color gating and removes them from the frame by replacing them with the clean background.

**Use Cases:**
- Magic-style invisibility cloak demos  
- Color-driven content masking  
- Low-cost video effects with static scenes  

> ✅ Default HSV range targets purple hues. Adjust it via the `--hsv_range` flag.

Example:

```bash
python -m apps.object_remove --src assets/demo_video/scene.mp4 --hsv_range purple 120 40 40 160 255 255 --mirror 0
````

---

## ⚙️ How It Works

1. **Input:** Loads a video file or webcam stream.
2. **Clean Plate Creation:** Manual capture or automatic median of N frames.
3. **Foreground Detection:** Color gating (HSV), frame differencing, or both.
4. **Mask Generation:** Binary mask highlighting areas to remove.
5. **Object Removal:** Masked areas filled with background plate or inpainted.
6. **Visualization:** Real-time output using various view modes.

---

## 🗂️ Project Structure

```
cv_lib/
  apps/
    object_remove.py         # Main script
  modules/
    ...                      # Detection, background, fill utils, etc.
  assets/
    demo_video/              # Test videos
    images/                  # Screenshots and masks
  ml/
    ...                      # (Optional) models or tools
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/background-object-removal-demo-cv.git
cd background-object-removal-demo-cv
```

### 2. Install Dependencies

Ensure Python 3.7+ is installed.

```bash
pip install opencv-python numpy
```

> Add more dependencies as needed (e.g., `scikit-image`, `shapely`, etc.)

---

### 3. Run the Demo

#### Webcam Mode:

```bash
python -m apps.object_remove --mirror 1
```

#### With Video Input & Auto Plate:

```bash
python -m apps.object_remove --src assets/demo_video/cars.mp4 --auto_frames 30 --mirror 0
```

#### With Color-Based Removal (e.g., purple cloth):

```bash
python -m apps.object_remove --src assets/demo_video/scene.mp4 --hsv_range purple 120 40 40 160 255 255 --mirror 0
```

---

## 💡 Usage Tips

* Press `p` to capture a clean plate manually
* Press `s` to save the background plate
* Use `--view triptych` to show side-by-side comparison
* Use `--mask_mode`, `--fill_mode`, and `--hsv_range` for full control

---

## ✍️ Customization

* Modify color ranges for other cloths (red, green, etc.)
* Add live HSV sliders for fine-tuning
* Combine with object tracking for dynamic removal

---

## 🙏 Acknowledgments

* Developed with [OpenCV](https://opencv.org/) and Python
* Demo assets sourced from public/open datasets

---

*Explore the codebase and modules for extensible components and live debugging.*
