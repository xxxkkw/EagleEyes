import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from train import TrajLSTM, TrajectoryDataset


class TrajectoryPredictor:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path)
        self.model = TrajLSTM().eval()
        self.model.load_state_dict(checkpoint['model_state'])
        self.mean = checkpoint['mean']
        self.std = checkpoint['std']
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, observed_traj, predict_steps=10):
        # 预处理
        norm_traj = (observed_traj - self.mean) / self.std
        seq = torch.FloatTensor(norm_traj).unsqueeze(0).to(self.device)

        predictions = []
        current_seq = seq.clone()
        for _ in range(predict_steps):
            with torch.no_grad():
                next_step = self.model(current_seq).cpu().numpy()
                predictions.append(next_step[0])

                # 更新输入序列
                current_seq = torch.cat([
                    current_seq[:, 1:],
                    torch.FloatTensor(next_step).unsqueeze(0).unsqueeze(0).to(self.device)
                ], dim=1)

        # 反标准化
        pred_traj = np.array(predictions) * self.std + self.mean
        return pred_traj


def plot_trajectory(observed, predicted, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(observed[:, 0], observed[:, 1], 'b-o', label='Observed')
    plt.plot(predicted[:, 0], predicted[:, 1], 'r--s', label='Predicted')
    plt.title("Trajectory Prediction")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Predict and plot badminton trajectory.")
    parser.add_argument('--model_path', type=str, default="./saved_models/model_epoch500.pt")
    parser.add_argument('--trajectory_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="./prediction_results")
    parser.add_argument('--observed_len', type=int, default=30, help='Number of observed frames.')
    parser.add_argument('--predict_steps', type=int, default=10, help='Number of prediction steps.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 初始化预测器
    predictor = TrajectoryPredictor(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    # 从指定的CSV文件加载轨迹
    sample_traj = np.loadtxt(args.trajectory_file, delimiter=',', skiprows=1)[:, 1:]
    observed = sample_traj[:args.observed_len]  # 取指定数量的帧作为观测

    # 预测指定数量的后续帧
    predicted = predictor.predict(observed, predict_steps=args.predict_steps)

    # 可视化并保存
    output_path = os.path.join(args.output_dir, "prediction.png")
    plot_trajectory(observed, predicted, output_path)

    print(f"Prediction saved at {output_path}")
