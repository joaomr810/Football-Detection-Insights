# Football Player Detection & Tactical Insights

This project aims to develop a computer vision pipeline capable of detecting football players from a real video and extracting meaningful tactical insights. It is structured into three main phases:

- **Player Detection**: Applying object detection models to identify players across video frames.
- **Perspective Transformation**: Converting player coordinates to a top-down (2D) pitch view using homography.
- **Insights Extraction**: Generating visual and statistical metrics to analyze spatial patterns and team behavior.

---

## 🎥 Input Video

- Duration: ~8 minutes
- Resolution: 1920x1080 (Full HD)
- FPS: 30
- Camera: Fixed, lateral angle
- Format: `./input/video.mp4`

The match is a real 7-a-side football game. Due to the lateral camera position, players farther from the lens may be harder to detect consistently.

---

## 🔍 Player Detection

Two detection approaches based on **YOLOv8m** were tested:

### 1. Standard YOLOv8 Model
- Pre-trained on **COCO dataset** (80 classes including "person").
- Implemented via Ultralytics library.
- Suitable for general object detection, but often detects people on the sidelines or crowd.

### 2. Custom YOLOv8m Model
- Trained on a football-specific dataset from [Roboflow Universe](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi).
- Dataset: ~320 annotated football images with players, referees, ball and goalkeepers.
- Focused on classes: `player`, `goalkeeper`
- Trained for 25 epochs. Evaluation metrics are included in Annex B of the final report.

**Result**: The custom model outperformed the standard model in both number and accuracy of player detections. It was used for the full video analysis.

---

## 🧭 Perspective Transformation

Using `cv2.findHomography`, a homography matrix was computed by manually annotating the four pitch corners in a reference frame. This allowed transforming player coordinates into a top-down field representation.

- Pitch template: [`mplsoccer`](https://mplsoccer.readthedocs.io/)
- Projection quality: Generally accurate, with minor bias due to camera angle.
- Validation: Example projections shown in Annex D.

---

## 📊 Tactical Insights

Based on transformed coordinates, the following metrics were computed:

### 🔥 Heatmap of Player Presence
- Visualizes player density across the pitch.
- Strong central concentration observed.

### 📍 Zone Occupation
- Pitch divided into 3 vertical zones: Defense / Midfield / Attack.
- Majority presence found in attacking and midfield zones.

### 📈 Occupied Area (Convex Hull)
- Measures team spread per frame using convex hull.
- Average area: ~2026 m² → indicates compact team structure.

### ↔️ Lateral Spread
- Difference between minimum and maximum lateral (y-axis) positions per frame.
- Average spread: ~48 meters → suggests limited width usage.

---

## ⚠️ Limitations

- **Occlusion issues**: Detection drops during set pieces (corners, throw-ins).
- **Perspective bias**: Transformation accuracy reduces in zones far from the camera.
- **No team differentiation**: All players are analyzed jointly, without team separation.

---

## 🚀 Future Improvements

- Continue training the custom model to increase accuracy.
- Add temporal smoothing or short-term tracking to stabilize player counts across frames.
- Improve transformation using more reference points or automated calibration.
- Use clustering (e.g., K-Means) on bounding box color features to differentiate teams.

---

## 🧰 Tech Stack

- Python (Jupyter + `.py` modules)
- OpenCV
- Ultralytics / YOLOv8
- mplsoccer
- NumPy / Matplotlib / SciPy

---

## 📁 Project Structure
