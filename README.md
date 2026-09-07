# YOLO 数据集工具箱 (YOLO Dataset Toolkit)

> 一站式 YOLO 检测数据集制作与训练 GUI 工具 —— 从视频到模型的完整闭环

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-tkinter-009688)
![Framework](https://img.shields.io/badge/ultralytics-8.4-7B1FA2)
![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-blue)
![Version](https://img.shields.io/badge/Version-3.1-brightgreen)

一个覆盖 **视频抽帧 → 数据划分 → 标注 → 训练 → 半自动预标注 → 微调 → 最终划分 → 二次训练** 完整流程的图形化工具。适合需要反复制作和训练 YOLO 检测数据集的算法工程师，无需手写脚本、手动改路径。

## 特性

- 🖥️ **图形界面**：7 个页签对应完整工作流，中文界面，零代码操作
- 📹 **视频抽帧**：按每秒 N 帧采样，支持**相似帧去重**（跳过几乎相同的帧，减少重复标注量），自动处理中文文件名/路径
- 🏷️ **半自动标注**：先用少量数据训练初版模型，再自动预测剩余图片生成标注框，人工只需微调
- 🤖 **一键训练**：**多模型排队对比训练**（勾选多个模型顺序训练，结束后自动输出 mAP 对比汇总表）；训练参数图形化配置——基础参数（epochs/imgsz/batch 等）+ **高级选项**（点开可调学习率 lr0/lrf、数据增强 mosaic/fliplr/degrees、权重衰减、cos_lr、device、seed，每个参数带 **❓ 悬停详细中文介绍**，留空用默认值）；后台子进程运行，可随时停止
- 🗂️ **数据集管理**：20% 抽取、70/30、70/20/10 划分自动完成；**无检出图片可选当背景图纳入**（提升精确率）；防呆设计防误操作
- 💾 **参数记忆**：所有路径与参数自动保存，项目根目录记住最近 5 个，下次打开不用重填
- 📊 **数量实时预览**：设置比例时立即显示预计张数
- 🌏 **中文路径安全**：解决 OpenCV 在 Windows 中文路径下静默失败的问题

## 工作流

```
①视频抽帧 → ②抽取20% → ③labelImg标注 → ④第一次训练
     ↓
⑤预测剩余80%生成标签 → ⑥微调+最终划分(70/20/10) → ⑦第二次训练
```

## 快速开始

### 环境要求

- Windows 10/11
- Python 3.11（需有 NVIDIA GPU 训练才快）

### 安装

```bash
# 1. 创建 conda 环境
conda create -n yolo python=3.11 -y
conda activate yolo

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装标注工具 labelImg（独立环境可选）
conda create -n labelimg -y
conda activate labelimg
pip install labelimg
```

### 运行

```bash
# 方式一：双击 启动工具.bat（自动探测 python 环境，无需配置）
# 方式二：命令行
conda activate yolo
python yolo_tool.py
```

> 环境路径全自动探测（优先名为 `yolo` 的 conda 环境），clone 下来即可运行；
> 探测失败时 GUI 启动日志会给出明确指引，手动修改 config.json 即可。
>
> ⚠️ 若预训练权重（yolov8n.pt 等）不在工具目录，ultralytics 会自动联网下载；
> 无法联网的环境需手动把 .pt 文件放入工具目录。

### 使用步骤

1. 顶部选择**项目根目录**（空文件夹，所有产物自动组织在其下）
2. 页签① 选视频文件夹 → 设置每秒帧数 → 抽帧
3. 页签② 设置抽取比例/类别名 → 执行抽取（自动生成 train/val 和 classes.txt）
4. 页签③ 打开 labelImg 标注 train / val 图片
5. 页签④ 配置模型与超参数 → 开始第一次训练
6. 页签⑤ 刷新选择 best.pt → 预测剩余图片（txt 自动回填）
7. 页签⑥ 微调预测框 → 执行最终划分（70/20/10 复制模式，原件不动）
8. 页签⑦ 对最终数据集进行第二次训练

## 项目结构

```
├── yolo_tool.py        # GUI 主程序（tkinter，7页签）
├── tool_core.py        # 核心逻辑层（抽帧/划分/训练/预测，可独立命令行使用）
├── 启动工具.bat         # Windows 一键启动脚本
├── TECH_DOC.md         # 技术交接文档（架构、协议、踩坑记录）
├── requirements.txt    # Python 依赖
└── config.json         # 运行时自动生成（路径/参数记忆）
```

## 技术架构

双进程模型：GUI（tkinter）负责交互与轻量文件操作；抽帧/训练/预测等重活在**子进程**中运行，stdout 流式回传日志，互不阻塞、崩溃隔离。详见 [TECH_DOC.md](TECH_DOC.md)。

## 目录产物约定

以项目根 `P:\` 为例，工具自动创建：

```
P:\
├── images_all/       # 抽帧输出（剩余图片供预测）
├── dataset/          # 手工标注集（20%）
├── rest_labels/      # 预测标签回填
├── final_dataset/    # 最终数据集（70/20/10）
├── predict_out/      # 预测可视化输出
└── result/           # 训练 runs（best.pt 等）
```

## 常见问题

| 问题 | 解决 |
|---|---|
| 训练报 ConnectionError | 权重文件不在本地且无法联网，把 .pt 拷入工具目录 |
| labelImg 打不开 | 页签③点「自动检测路径」重新定位 |
| 显存不足 | 调小 batch 或 imgsz |
| 运行产物误删 | config.json 会自动备份（.bak） |

## 许可证

[MIT](LICENSE)

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [labelImg](https://github.com/HumanSignal/labelImg)
