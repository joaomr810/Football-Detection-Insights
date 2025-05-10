import cv2
import os
import torch
from tqdm.notebook import tqdm
from ultralytics import YOLO


class DetectionYOLO:
    """
    A class for performing person detection on a video using a YOLO model, 
    saving per-frame detection results, and generating an annotated output video.

    Args:
        input_video_path (str): Path to the input video file
        output_video_path (str): Path to save the annotated output video
        detections_path (str): Directory where detection text files will be saved
        model_path (str): Path to the YOLO model weights
        minutes (int or float): Maximum number of minutes of video to process
    """

    def __init__(self, input_video_path, output_video_path, detections_path, model_path, minutes):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.detections_path = detections_path
        self.model_path = model_path
        self.minutes = minutes

        os.makedirs(self.detections_path, exist_ok=True)

        self.cap = cv2.VideoCapture(self.input_video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.max_frames = int(self.minutes * 60 * self.fps)

        self.model = YOLO(self.model_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(self.output_video_path, fourcc, self.fps,
                                            (self.frame_width, self.frame_height))

    def run(self):
        """
        Executes the detection process on the input video.
        Reads the video frame by frame, applies the YOLO model for detection,
        saves detections in MOT format, annotates frames, and writes an output video.       
        """
        if not self.cap.isOpened():
            print('Error: Could not open video file.')
            return

        print('Video file opened successfully!')
        print(f'- Processing up to {self.max_frames} frames (~{self.minutes} minute(s)).\n')

        frame_num = 0

        with tqdm(total=self.max_frames, desc='Person Detection') as pbar:
            while self.cap.isOpened() and frame_num < self.max_frames:
                ret, frame = self.cap.read()
                if not ret:
                    break

                with torch.no_grad():
                    results = self.model(frame)[0]

                with open(os.path.join(self.detections_path, f'{frame_num:06d}.txt'), 'w') as f:
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        w, h = x2 - x1, y2 - y1
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())

                        if cls_id == 0 and conf >= 0.5:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f'Conf: {conf:.2f}', (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                            f.write(f'{frame_num}, -1, {x1:.2f}, {y1:.2f}, {w:.2f}, {h:.2f}, {conf:.4f}, -1, -1\n')

                self.output_video.write(frame)
                frame_num += 1
                pbar.update(1)

        self.cap.release()
        self.output_video.release()

        print(f'\n- Output video with detections saved at {self.output_video_path}')
        print(f'- Detections saved at: {self.detections_path}')


class DetectionCustom:
    """
    A class for performing detection of classes 0 and 1 on a video using a YOLO model, 
    saving per-frame detection results, and generating an annotated output video.

    Args:
        input_video_path (str): Path to the input video file
        output_video_path (str): Path to save the annotated output video
        detections_path (str): Directory where detection text files will be saved
        model_path (str): Path to the YOLO model weights
        minutes (int or float): Maximum number of minutes of video to process
    """
    
    def __init__(self, input_video_path, output_video_path, detections_path, model_path, minutes):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.detections_path = detections_path
        self.model_path = model_path
        self.minutes = minutes

        os.makedirs(self.detections_path, exist_ok=True)

        self.cap = cv2.VideoCapture(self.input_video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.max_frames = int(self.minutes * 60 * self.fps)

        self.model = YOLO(self.model_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(self.output_video_path, fourcc, self.fps,
                                            (self.frame_width, self.frame_height))

    def run(self):
        """
        Executes the detection process on the input video.

        Reads the video frame by frame, applies the detection model for selected classes,
        saves detections in MOT format, annotates frames, and writes an output video.        
        """

        if not self.cap.isOpened():
            print('Error: Could not open video file.')
            return

        print('Video file opened successfully!')
        print(f'- Processing up to {self.max_frames} frames (~{self.minutes} minute(s)).\n')

        frame_num = 0

        with tqdm(total=self.max_frames, desc='Player Detection') as pbar:
            while self.cap.isOpened() and frame_num < self.max_frames:
                ret, frame = self.cap.read()
                if not ret:
                    break

                with torch.no_grad():
                    results = self.model(frame)[0]

                with open(os.path.join(self.detections_path, f'{frame_num:06d}.txt'), 'w') as f:
                    for box in results.boxes:
                        conf = box.conf[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())

                        if cls_id in [1, 2] and conf >= 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            w, h = x2 - x1, y2 - y1
                            label = self.model.names[cls_id]

                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                            f.write(f'{frame_num}, -1, {x1:.2f}, {y1:.2f}, {w:.2f}, {h:.2f}, {conf:.4f}, -1, -1\n')

                self.output_video.write(frame)
                frame_num += 1
                pbar.update(1)

        self.cap.release()
        self.output_video.release()

        print(f'\n- Output video with detections saved at {self.output_video_path}')
        print(f'- Detections saved at: {self.detections_path}')