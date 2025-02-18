import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class TrajectoryDataset(Dataset):
    def __init__(self, data_folder, seq_length=30, predict_steps=10):
        self.sequences = []
        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

        for f in csv_files:
            data = np.loadtxt(os.path.join(data_folder, f), delimiter=',', skiprows=1)
            coords = data[:, 1:]

            # 标准化
            self.mean = np.mean(coords, axis=0)
            self.std = np.std(coords, axis=0)
            norm_coords = (coords - self.mean) / self.std

            # 生成序列
            for i in range(len(norm_coords) - seq_length - predict_steps):
                seq = norm_coords[i:i + seq_length]
                target = norm_coords[i + seq_length:i + seq_length + predict_steps]
                self.sequences.append((seq, target))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.FloatTensor(seq), torch.FloatTensor(target)


class TrajLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out


def evaluate(model, dataloader, device):
    model.eval()
    total_error = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            preds = model(inputs)

            # 反标准化
            preds = preds.cpu().numpy() * dataset.std + dataset.mean
            targets = targets.numpy() * dataset.std + dataset.mean

            # 计算相对误差
            error = np.mean(np.linalg.norm(preds - targets, axis=1) / np.linalg.norm(targets, axis=1))
            total_error += error
    return total_error / len(dataloader)


def train(args):
    # 数据集
    full_dataset = TrajectoryDataset(args.data_dir, args.seq_len, args.pred_steps)
    train_data, test_data = train_test_split(full_dataset, test_size=0.2, shuffle=False)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    # 模型
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = TrajLSTM(hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    train_loss = []
    test_errors = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets[:, 0, :])  # 预测第一个点
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 每50轮评估
        if (epoch + 1) % 50 == 0:
            test_error = evaluate(model, test_loader, device)
            test_errors.append(test_error)
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}, Test Error: {test_error:.4f}")

            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'mean': full_dataset.mean,
                'std': full_dataset.std
            }, f"{args.save_dir}/model_epoch{epoch + 1}.pt")

        train_loss.append(epoch_loss / len(train_loader))

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.title("Training Loss Curve")
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(50, args.epochs + 1, 50), test_errors, 'r-', label='Test Error')
    plt.title("Test Relative Error")
    plt.savefig(f"{args.save_dir}/training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="trajectories")
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--pred_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)