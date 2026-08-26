# YOLO 数据集工具箱 — 技术交接文档

| | |
|---|---|
| **版本** | 2.0（2026-08-26）—— 多模型对比/去重/背景图/便利功能 |
| **作者** | muwen（由 Claude Code 协助开发） |
| **用途** | 一站式 YOLO 检测数据集制作 + 训练 GUI 工具 |
| **技术栈** | Python 3.11 / tkinter / ultralytics 8.4.49 / OpenCV 4.11 |

---

## 1. 快速上手

```
双击 启动工具.bat
→ 顶部选择"项目根目录"（一个空文件夹，如 D:\00software\proj001）
→ 按 ①~⑦ 页签顺序操作即可
```

所有路径参数自动记忆（`config.json`），下次打开不用重填。

## 2. 文件清单

```
D:\00software\新建文件夹 (2)\
├── yolo_tool.py        GUI 主程序（tkinter，7页签）
├── tool_core.py        核心逻辑层（可独立命令行运行）
├── 启动工具.bat         双击启动器（用 conda yolo 环境拉起 GUI）
├── config.json         运行时自动生成的配置（GUI 同目录）
├── yolov8n.pt          预训练权重（本地化，离线可用）
├── yolov8s.pt
└── yolo11n.pt
```

> ⚠️ 权重文件是刻意复制到本目录的：训练环境无法访问 GitHub，
> ultralytics 找不到权重时会尝试联网下载并失败。换机器部署时
> 从 `D:\00software\ultralytics-8.4.67\` 再拷贝一份即可。

## 3. 用户工作流（7 个页签）

这是用户多年习惯的流程，工具完全按此设计，**不要随意改动顺序语义**：

```
① 视频抽帧      视频文件夹 --按每秒N帧--> <根>/images_all/*.jpg
② 抽取20%       随机【移动】20% → <根>/dataset/images/{train,val} (内部70/30)
                同时创建 dataset/labels/{train,val} + classes.txt
③ 启动标注      自动拉起 labelImg 加载 train / val 图片目录
④ 第一次训练    自动生成 dataset.yaml → 子进程训练 → result/<模型名>/weights/best.pt
⑤ 预测剩余80%   用 best.pt 对 images_all 剩余图片预测(save_txt)
                → txt 自动回填 <根>/rest_labels/
⑥ 微调+最终划分 labelImg 微调预测框 → 【复制】划分为 final_dataset/{images,labels}/{train,val,test}
                = 70/20/10（test 取余数），无标签图跳过并列清单
⑦ 第二次训练    同④，数据集指向 final_dataset
```

关键用户决策（改需求前先确认）：
- ② 是**移动**不是复制（剩余 80% 留原地供⑤预测）
- 先分 train/val 再标注（labelimg 要分别打开两次）
- 最终数据集**只用微调后的 80%**，不合并最初手工标注的 20%

## 4. 架构

### 4.1 双进程模型

```
启动工具.bat ──> python.exe yolo_tool.py   ← GUI 进程（tkinter 主线程）
                    │
                    ├── 轻量任务（抽取20%/最终划分/yaml生成）:
                    │     threading.Thread 内直接调 core 函数
                    │     通过 core.log_hook 钩子回收日志
                    │
                    └── 重活（抽帧/训练/预测）:
                          subprocess.Popen([python_exe, tool_core.py, --job ...])
                          stdout 逐行 → queue.Queue → root.after(100ms) 轮询
                          → App._on_job_line 更新日志区/状态栏
                          → App._on_job_exit 恢复按钮状态

labelImg 启动 = fire-and-forget Popen，不纳入互斥管理
```

为什么这样设计：
- **子进程隔离**：ultralytics 训练吃 GPU 且不可中断 GUI 主线程；崩溃不拖垮界面
- **不用 multiprocessing**：Windows spawn 会 re-import 主模块，跨进程传 tkinter 对象是坑
- **轻量任务不走子进程**：秒级文件操作，走子进程反而要造进度协议

### 4.2 子进程协议

命令行格式（`tool_core.py main()` 的 argparse）：

```
--job extract  --video_dir X --out_dir Y --fps 5
--job train    --data yaml路径 --model yolov8n.pt --project_root 根
               --epochs 300 --imgsz 224 --batch 32 --patience 50
               --copy_paste 0.3 --workers 2 [--cache_ram 1]
--job predict  --weights best.pt --source 目录 --conf 0.25
               --project <根>/predict_out --name predict1
```

stdout 行协议：

| 行 | 含义 | GUI 处理 |
|---|---|---|
| 普通行 | 日志 | 追加日志区 |
| `PROGRESS cur/total` | 抽帧进度 | 更新状态栏 |
| `SAVE_DIR: 路径` | 任务产物目录 | 记录，成功后展示 |
| `JOB_ERROR: 中文消息` | 业务失败 | 红色显示，exit code 2 |
| exit 0/1/2 | 成功/异常/业务失败 | `_on_job_exit` 分支处理 |

环境变量：子进程注入 `PYTHONIOENCODING=utf-8`（GBK 控制台防乱码）、`KMP_DUPLICATE_LIB_OK=TRUE`（防 OpenMP 崩溃）。

### 4.3 关键类

| 类 | 位置 | 职责 |
|---|---|---|
| `App(tk.Tk)` | yolo_tool.py | 全部 UI 与事件；`start_buttons` 列表实现任务互斥；多模型批次状态 `_train_batch` |
| `JobRunner` | yolo_tool.py | Popen 生命周期 + **任务队列**（`pending` 列表，串行执行）；`stop()` 终止当前并清空队列；`last_elapsed` 记录单任务耗时；`stopped_by_user` 区分人为停止与失败 |
| `Config` | yolo_tool.py | config.json 读写；损坏时备份 `.bak` 后重建默认；v1→v2 自动迁移（`model` 单值 → `models` 列表） |

`tool_core.py` 顶层只 import 标准库——cv2/numpy/ultralytics 在各函数内部延迟导入，保证 GUI 进程零第三方依赖。

### 4.4 JobRunner 队列协议（v2.0）

多模型排队 = GUI 层 for 循环逐个 `runner.start(cmd, desc, on_done)`：
- 空闲时立即启动第一个，其余入 `pending` 队列；上一个退出后自动续跑
- `desc` 格式约定 `"训练 {模型名} (i/N)"`，App 用 `runner.desc` 关联当前批次项
- **退出码读取时机**：stdout EOF 后必须 `proc.wait(timeout=10)` 再取码——CUDA 清理可能延后于管道关闭，直接 `poll()` 会误报 -1 失败（v2.0 修复）
- 用户 Stop：终止当前进程 + 清空剩余队列，批次标记停止
- 全部完成后 App 读取各 run 的 `results.csv` 末行输出对比表（`core.read_results_summary`）

## 5. 必须知道的坑（都是实测踩过的）

### ① cv2 中文路径静默失败 ⚠️ 最重要
Windows 下 `cv2.imwrite/imread` 遇中文路径**不报错但什么都不做**。
必须用封装（tool_core.py 的 `save_frame` / `read_image`）：
```python
cv2.imencode('.jpg', frame)[1].tofile(path)      # 写
np.fromfile(path, np.uint8) → cv2.imdecode(...)  # 读
```
用户的目录全是中文名，绕过封装直接用 imwrite 整个工具就废了。

### ② 系统 PATH 的 python 不是训练环境（v1.1 已自动探测）
`python` 命令可能指向不带 ultralytics 的 Python。
v1.1 起 `core.find_python_exe()` 自动探测解释器，优先级：
1. `CONDA_EXE` 推断的 `envs/yolo`
2. 常见 conda 安装位置的 `envs/yolo`（含 D:\ 盘位 glob）
3. 当前进程解释器 `sys.executable`（GUI 用什么环境启动就复用哪个，pythonw 自动转 python.exe）
4. conda base

探测结果写入 config.json 的 `python_exe`；GUI 启动时在日志区展示。
`validate_python_exe()` 会在训练前**实际运行 `import ultralytics` 验证**（带缓存），
环境不对会给出明确报错而不是等训练才失败。

### ③ labelImg 不在 PATH（v1.1 已自动探测）
`core.find_labelimg_exe()` 探测链：config 记录路径 → CONDA_EXE 推断 +
常见 conda 安装位置（含 D:\00software\anaconda）下扫描各 env 的 Scripts → shutil.which。
找不到时 GUI 启动日志会提示，可在页签③「自动检测路径」或手动浏览选择。

### ④ ultralytics save_txt 只为有检出的图生成 txt
没检出目标的图片没有 txt 文件——这是正常行为不是 bug。
v2.0 起页签⑥可选「无检出图片当背景图纳入」：生成 0 字节空 txt 一并复制进
final_dataset（YOLO 视空标签为纯背景负样本）；风险：漏检的目标会被教成背景。

### ⑤ tqdm 进度条刷屏
ultralytics 训练输出带 `%|` 的 `\r` 刷新行，`_poll()` 里已过滤只保留有效日志。

### ⑥ GBK 控制台 UnicodeEncodeError
独立命令行跑 tool_core.py 时中文符号可能打不出，`log()` 已做 gbk replace 降级，不会炸任务。

### ⑦ 退出码误报（v2.0 已修复）
子进程 stdout 关闭（EOF）早于进程真正退出（CUDA 清理延后），EOF 后直接 `poll()` 返回
None → 误报 exit=-1「失败」。必须 `wait(timeout=10)` 等进程结束再取码（见 §4.4）。

## 6. 运行时产物目录结构

以项目根 `P:\` 为例（全部由 GUI 自动创建）：

```
P:\
├── images_all/                 ①抽帧输出；②之后剩余的图片留在这里供⑤预测
├── dataset/
│   ├── images/{train,val}/     ②移动进来的手工标注集(20%)
│   ├── labels/{train,val}/     ③labelImg 标注落盘处
│   ├── classes.txt             ②生成；类别名每行一个 ← yaml names 的唯一事实源
│   └── dataset.yaml            ④每次开训自动重建
├── rest_labels/                ⑤预测txt自动回填到这里（与 images_all 同级配对）
├── predict_out/predict1/       ⑤原始预测输出（可视化jpg + labels/*.txt）
├── final_dataset/              ⑥复制划分产物（原件不动）
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── final_dataset.yaml      ⑦每次开训自动重建
└── result/<模型名>/            ④⑦训练 runs（weights/best.pt、results.png 等）
```

## 7. 防呆设计（维护时别拆掉）

| 场景 | 行为 |
|---|---|
| 重复执行抽取20% | 报错拒绝（防止两批混合污染） |
| 已有标注后修改类别名 | 报错拦截（索引错位会污染全部标签） |
| 移动/划分执行前 | 弹窗显示具体数量二次确认 |
| 输出目录已有内容 | 报错要求先清理（保护已标注成果） |
| 任务运行中关窗 | askokcancel 确认后终止任务退出 |
| 比例和 >100% | 点击即弹错，不发任务 |

## 8. 测试情况（v2.0 全量实测通过）

- 语法编译 ✅；核心函数单测 ✅
- 抽帧：中文名视频 @25fps 源 @5fps 抽取 → 15 张正确命名落盘 ✅
- 抽取20%：100张 → 14/6/80，防呆三连（重复执行/类名变更/比例非法）✅
- 划分：70对 → 49/14/7，缺标签跳过报告，原件未动 ✅
- 微型真实训练（yolov8n, 1 epoch, 64px）→ best.pt 落盘 ✅
- 预测链路 → predict_out 结构正确，空检出时回填 0 属预期 ✅
- JobRunner 子进程：流式日志、退出回调、Stop 终止（stopped_by_user=True）✅
- config.json 往返持久化 ✅；GUI 冒烟（7页签构建）✅
- **v2.0** 多模型排队（2模型各1epoch 实测31~36s 串行完成，exit 全 0）✅
- **v2.0** 对比汇总表（mAP50/mAP50-95/用时，失败行标注"失败/未完成"）✅
- **v2.0** 相似帧去重（75帧视频 → 去重后仅2张，阈值99%）✅
- **v2.0** 背景图纳入（10图6标4空 → 全部进 final_dataset 且 label 配对）✅
- **v2.0** v1→v2 config 迁移（model 单值→models 列表）、最近项目历史、数量预览 ✅

## 9. 已知限制 / 未做功能

- 单项目单任务：同一时刻只能跑一个**任务队列**（多模型排队算一个队列，内部串行）
- `cache=ram` 默认关：大数据集会 OOM，勾选前确认内存充足
- imgsz 默认 224 是用户习惯值（小目标场景可自行调大）
- 多模型对比仅报告 mAP 数值，不自动绘制对比图（结果目录有各模型独立曲线，可自行比对）
- 相似帧去重阈值 100% 时行为 ≈ 不去重（仅跳过完全相同帧）

## 10. 常见问题排查

| 症状 | 排查方向 |
|---|---|
| 训练一开就 ConnectionError | 权重不在本目录且无法联网 → 把 .pt 拷进工具目录 |
| 日志区乱码 | 不影响功能；子进程已强制 UTF-8，仅控制台直跑会出现 |
| labelImg 点了没反应 | 页签③「自动检测路径」重新定位 exe |
| 抽帧 0 张输出 | 检查视频编码是否被 OpenCV 支持（手机视频一般没问题）；看日志有无「无法打开视频」警告 |
| 训练中途想停 | 对应页签「停止训练」按钮；terminate 后 3 秒强杀 |
| config.json 坏了 | 删掉重启即可，程序自动备份损坏文件为 .bak 并重建默认 |

## 11. 部署到新机器（v1.1 起基本零配置）

1. 安装 Miniconda/Anaconda，创建环境并装包（**环境名建议用 yolo**，自动探测默认找它）：
   ```bash
   conda create -n yolo python=3.11 -y
   conda activate yolo
   pip install ultralytics opencv-python
   ```
2. 另建 labelimg 环境：`pip install labelimg`（环境名随意，自动探测会扫所有 envs）
3. 拷贝整个工具文件夹；把预训练 .pt 权重放进同目录
4. 双击 `启动工具.bat` —— 自动探测 python（找不到才需手动改 bat 候选路径）；
   GUI 启动时自动探测 python_exe 与 labelimg_exe 并写入 config.json，日志区可见
5. 若自动探测失败（环境装在奇怪位置）：手动改 `config.json` 的
   `python_exe` / `labelimg_exe`，或页签③「自动检测路径」

> 约定：探测默认找名为 **yolo** 的 conda 环境；不想改环境名的话
> 直接手动配 config.json 即可，探测结果不会覆盖用户已填写的路径。

## 12. Git 仓库说明

- 仓库位置即工具目录；`.gitignore` 排除 `__pycache__/`、`*.pt`（大文件不入库，见 §2 部署说明）、`config.json`（含个人路径）
- tag 历史：`v1.0`（基础版）→ `v1.1`（环境自动探测）→ `v2.0`（多模型对比/去重/背景图）
