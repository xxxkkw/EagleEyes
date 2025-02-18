# 羽毛球轨迹预测系统 (Badminton Trajectory Prediction)

基于YOLOv5目标检测与LSTM时序建模的羽毛球运动轨迹预测系统

## 🚀 功能特性

- **高精度检测**: 采用YOLOv5实时检测羽毛球位置
- **轨迹建模**: 使用深度LSTM网络学习运动模式
- **可视化分析**: 生成轨迹对比图与训练曲线
- **多步预测**: 支持任意长度的轨迹预测



## 📦 环境要求

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
 ```

## 🗂 项目结构
```plaintext
EagleEyes/
├── input_videos/          # 原始视频存储目录
├── trajectories/          # 提取的轨迹CSV文件
├── saved_models/          # 训练好的模型参数
├── prediction_results/    # 预测结果可视化
├── extract.py             # 轨迹提取脚本
├── train.py               # LSTM训练脚本
└── predict.py             # 预测与可视化脚本
```
## 🛠 使用指南
1. 轨迹提取
```bash
python extract.py \
    --video_dir input_videos \
    --output_dir trajectories \
    --visualize 
```

2. 模型训练
```bash
python train.py \
    --data_dir ./trajectories \
    --save_dir ./saved_models \
    --epochs 500 \
    --batch_size 64
```

3. 轨迹预测
```bash
python predict.py \
    --model_path ./saved_models/model_epoch500.pt \
    --trajectory_file ./trajectories/test1.csv \
    --output_dir ./prediction_results
```
## ⚙ 参数说明
### 训练参数 (train.py)
| 参数名         | 默认值 | 说明               |
|----------------|--------|--------------------|
| --seq_len      | 30     | 输入序列长度（帧数） |
| --pred_steps   | 10     | 预测步长            |
| --hidden_size  | 128    | LSTM隐藏层维度      |
| --num_layers   | 3      | LSTM堆叠层数        |
### 预测参数 (predict.py)
| 参数名           | 默认值 | 说明               |
|------------------|--------|--------------------|
| --observed_ratio | 0.3    | 输入观测比例         |
| --predict_steps  | 10     | 预测帧数            |
| --line_width     | 2      | 可视化线条粗细      |
## 📊 结果示例

### 轨迹对比图
![prediction](path_to_image)

- **蓝色实线**: 实际观测轨迹
- **红色虚线**: 模型预测轨迹
### 训练曲线

![training_curves](path_to_image)

- 左: 训练损失下降曲线
- 右: 测试集相对误差
