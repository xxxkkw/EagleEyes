import os
import cv2
import csv
import torch
import argparse
from tqdm import tqdm
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression


class TrajectoryExtractor:
    def __init__(self, model_path, output_dir, visualize=False):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = attempt_load(model_path, device=self.device)
        self.output_dir = output_dir
        self.visualize = visualize
        os.makedirs(output_dir, exist_ok=True)

    def process_videos(self, video_dir):
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

        for vid_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(video_dir, vid_file)
            cap = cv2.VideoCapture(video_path)
            trajectory = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            with tqdm(total=total_frames, desc=f"Processing {vid_file}", unit="frame") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (640, 640))

                    # Preprocess the frame
                    img = torch.from_numpy(frame).permute(2, 0, 1).float().to(self.device)
                    img /= 255.0

                    # Model inference
                    pred = self.model(img[None])[0]
                    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

                    if len(pred[0]) > 0:
                        for det in pred[0]:
                            if det[5] == 0:
                                x1, y1, x2, y2 = det[:4].cpu().numpy()
                                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                                trajectory.append((cx, cy))

                                if self.visualize:  # 可视化开关
                                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)

                    # Display the frame with the trajectory points if visualization is enabled
                    if self.visualize:
                        cv2.imshow('Trajectory Visualization', frame)

                        # Wait for a short period to allow visualization, press 'q' to quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    pbar.update(1)

            output_path = os.path.join(self.output_dir, f"{os.path.splitext(vid_file)[0]}.csv")
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'x', 'y'])
                for i, (x, y) in enumerate(trajectory):
                    writer.writerow([i, x, y])

            cap.release()

        if self.visualize:
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract badminton trajectory from videos.")
    parser.add_argument('--model_path', type=str, required=False, default="badminton_best.pt")
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing input videos.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save extracted trajectory CSV files.')
    parser.add_argument('--visualize', action='store_true', help='Enable real-time visualization of the trajectory')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extractor = TrajectoryExtractor(args.model_path, args.output_dir, visualize=args.visualize)
    extractor.process_videos(args.video_dir)
