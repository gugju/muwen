# YOLO 数据集工具箱 — 技术交接文档

| | |
|---|---|
| **版本** | 1.0（2026-08-26） |
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
| `App(tk.Tk)` | yolo_tool.py | 全部 UI 与事件；`start_buttons` 列表实现任务互斥 |
| `JobRunner` | yolo_tool.py | Popen 生命周期；`stop()` 先 terminate 3 秒后 kill；`stopped_by_user` 区分人为停止与失败 |
| `Config` | yolo_tool.py | config.json 读写；损坏时备份 `.bak` 后重建默认 |

`tool_core.py` 顶层只 import 标准库——cv2/numpy/ultralytics 在各函数内部延迟导入，保证 GUI 进程零第三方依赖。

## 5. 必须知道的坑（都是实测踩过的）

### ① cv2 中文路径静默失败 ⚠️ 最重要
Windows 下 `cv2.imwrite/imread` 遇中文路径**不报错但什么都不做**。
必须用封装（tool_core.py 的 `save_frame` / `read_image`）：
```python
cv2.imencode('.jpg', frame)[1].tofile(path)      # 写
np.fromfile(path, np.uint8) → cv2.imdecode(...)  # 读
```
用户的目录全是中文名，绕过封装直接用 imwrite 整个工具就废了。

### ② 系统 PATH 的 python 不是训练环境
`python` 命令指向 Python 3.13，**没有 ultralytics**。
一切训练/预测子进程必须显式用 `D:\00software\anaconda\envs\yolo\python.exe`
（记录在 config.json 的 `python_exe`）。

### ③ labelImg 不在 PATH
实际位置 `D:\00software\anaconda\envs\labelimg\Scripts\labelImg.exe`。
`core.find_labelimg_exe()` 三级 fallback：config 记录路径 → 扫描各 conda env 的 Scripts → shutil.which。

### ④ ultralytics save_txt 只为有检出的图生成 txt
没检出目标的图片没有 txt 文件——这是正常行为不是 bug。
这些图在⑥最终划分时被跳过并列出清单（≤10 条明细）。

### ⑤ tqdm 进度条刷屏
ultralytics 训练输出带 `%|` 的 `\r` 刷新行，`_poll()` 里已过滤只保留有效日志。

### ⑥ GBK 控制台 UnicodeEncodeError
独立命令行跑 tool_core.py 时中文符号可能打不出，`log()` 已做 gbk replace 降级，不会炸任务。

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

## 8. 测试情况（2026-08-26 实测通过）

- 语法编译 ✅；核心函数单测 ✅
- 抽帧：中文名视频 @25fps 源 @5fps 抽取 → 15 张正确命名落盘 ✅
- 抽取20%：100张 → 14/6/80，防呆三连（重复执行/类名变更/比例非法）✅
- 划分：70对 → 49/14/7，缺标签跳过报告，原件未动 ✅
- 微型真实训练（yolov8n, 1 epoch, 64px）→ best.pt 落盘 ✅
- 预测链路 → predict_out 结构正确，空检出时回填 0 属预期 ✅
- JobRunner 子进程：流式日志、退出回调、Stop 终止（stopped_by_user=True）✅
- config.json 往返持久化 ✅；GUI 冒烟（7页签构建）✅

## 9. 已知限制 / 未做功能（经用户确认暂缓）

以下在需求讨论中确认过价值，但用户决定先交付 1.0，完整方案见
`C:\Users\muwen\.claude\plans\yolo-tool-plan-v2-enhancements.md`：

- 多模型排队对比训练 + mAP 汇总表（当前一次只能训一个模型）
- 相似帧去重开关（减少重复标注量）
- 无检出图片当背景图纳入的可选项（当前一律跳过）
- 数量实时预览、最近项目历史下拉
- 断点续训、完成提醒/自动关机、日志落盘（明确不做）

其他限制：
- 单项目单任务：同一时刻只能跑一个重活（互斥锁设计如此）
- `cache=ram` 默认关：大数据集会 OOM，勾选前确认内存充足
- imgsz 默认 224 是用户习惯值（小目标场景可自行调大）

## 10. 常见问题排查

| 症状 | 排查方向 |
|---|---|
| 训练一开就 ConnectionError | 权重不在本目录且无法联网 → 把 .pt 拷进工具目录 |
| 日志区乱码 | 不影响功能；子进程已强制 UTF-8，仅控制台直跑会出现 |
| labelImg 点了没反应 | 页签③「自动检测路径」重新定位 exe |
| 抽帧 0 张输出 | 检查视频编码是否被 OpenCV 支持（手机视频一般没问题）；看日志有无「无法打开视频」警告 |
| 训练中途想停 | 对应页签「停止训练」按钮；terminate 后 3 秒强杀 |
| config.json 坏了 | 删掉重启即可，程序自动备份损坏文件为 .bak 并重建默认 |

## 11. 部署到新机器

1. 安装 Miniconda/Anaconda，创建环境并装包：
   ```bash
   conda create -n yolo python=3.11 -y
   conda activate yolo
   pip install ultralytics opencv-python
   ```
2. 另建 labelimg 环境：`pip install labelimg`（或改用系统 PATH 能找到的安装方式）
3. 拷贝整个工具文件夹；把预训练 .pt 权重放进同目录
4. 修改 `启动工具.bat` 里的 `PY_EXE` 和 `config.json` 里的 `python_exe` / `labelimg_exe` 为新机器路径
5. 双击 bat 启动验证

## 12. Git 仓库说明

- 仓库位置即工具目录；`.gitignore` 排除 `__pycache__/`、`*.pt`（大文件不入库，见 §2 部署说明）、`config.json`（含个人路径）
- 首次提交 tag：`v1.0`
