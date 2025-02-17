import os
import cv2
import csv
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression


class TrajectoryExtractor:
    def __init__(self, model_path, output_dir):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = attempt_load(model_path, device=self.device)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_videos(self, video_dir):
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4'))]

        for vid_file in video_files:
            video_path = os.path.join(video_dir, vid_file)
            cap = cv2.VideoCapture(video_path)
            trajectory = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                img = torch.from_numpy(frame).permute(2, 0, 1).float().to(self.device)
                img /= 255.0
                pred = self.model(img[None])[0]
                pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

                if len(pred[0]) > 0:
                    for det in pred[0]:
                        if det[5] == 0:
                            x1, y1, x2, y2 = det[:4].cpu().numpy()
                            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                            trajectory.append((cx, cy))

            output_path = os.path.join(self.output_dir, f"{os.path.splitext(vid_file)[0]}.csv")
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'x', 'y'])
                for i, (x, y) in enumerate(trajectory):
                    writer.writerow([i, x, y])

            cap.release()


if __name__ == "__main__":
    extractor = TrajectoryExtractor(
        model_path="yolov5/yolov5s.pt",
        output_dir="./trajectories"
    )
    extractor.process_videos("./input_videos")