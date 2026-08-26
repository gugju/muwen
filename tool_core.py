#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO数据集工具箱 —— 核心逻辑层
=================================
版本: 2.0 (2026-08-26) —— 多模型对比训练/抽帧去重/背景图/便利功能
既可被 yolo_tool.py (GUI) 导入复用，也可独立命令行运行：

    python tool_core.py --job extract --video_dir X --out_dir Y --fps 5
    python tool_core.py --job train   --data Z.yaml --model yolov8n.pt ...
    python tool_core.py --job predict --weights best.pt --source DIR ...

注意：
- 顶层只允许 import 标准库；cv2/numpy/ultralytics 在各函数内部延迟导入，
  保证 GUI 进程导入本模块时零第三方依赖。
- 所有涉及中文路径的图像读写必须走 save_frame/read_image 封装
  （cv2.imwrite/imread 在 Windows 中文路径下会静默失败）。
"""

import argparse
import glob
import os
import random
import shutil
import sys
import time
import traceback

IMG_EXTS = (".jpg", ".jpeg", ".png")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")


class JobError(Exception):
    """带中文消息的业务错误，子进程以 JOB_ERROR 协议回传 GUI 并 exit(2)"""


# GUI 进程可设置该钩子接管日志输出（子进程模式保持 None 走 print）
log_hook = None


def log(msg):
    """打印日志。GBK 控制台打不出的字符降级替换，不让打印本身炸掉任务"""
    text = str(msg)
    if log_hook is not None:
        try:
            log_hook(text)
            return
        except Exception:
            pass
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode("gbk", errors="replace").decode("gbk"), flush=True)


# ---------------------------------------------------------------- #
# 通用小工具
# ---------------------------------------------------------------- #

def read_classes(classes_txt):
    """读取 classes.txt，每行一个类名，忽略空行。文件不存在返回空列表"""
    if not os.path.isfile(classes_txt):
        return []
    with open(classes_txt, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def list_images(dir_path):
    """列出目录下的图片文件名（不含子目录），排序返回"""
    if not os.path.isdir(dir_path):
        return []
    return sorted(
        f for f in os.listdir(dir_path)
        if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(dir_path, f))
    )


def find_labelimg_exe(configured=None):
    """
    自动探测 labelImg.exe（无需用户手动配置）：
      1. config 记录的路径存在则直接用
      2. CONDA_EXE 推断 + 常见 conda 安装位置的 Scripts 目录
      3. shutil.which 兜底
    都找不到返回 None
    """
    if configured and os.path.isfile(configured):
        return configured

    roots = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        roots.append(os.path.dirname(os.path.dirname(conda_exe)))
    for base in (os.path.expanduser("~"), r"C:\ProgramData", "D:\\", "C:\\"):
        for name in ("anaconda3", "miniconda3", "Anaconda3", "Miniconda3"):
            roots.append(os.path.join(base, name))
    roots.append(r"D:\00software\anaconda")

    for root in dict.fromkeys(roots):   # 去重保序
        for name in ("labelImg.exe", "labelimg.exe"):
            for p in glob.glob(os.path.join(root, "envs", "*", "Scripts", name)):
                if os.path.isfile(p):
                    return p
            p = os.path.join(root, "Scripts", name)
            if os.path.isfile(p):
                return p
    return shutil.which("labelImg") or shutil.which("labelimg")


def find_python_exe():
    """
    自动探测可用的 python.exe（无需用户手动配置），优先带 ultralytics 的 conda 环境：
      1. CONDA_EXE 推断的 envs/yolo
      2. 常见 conda 安装位置的 envs/yolo（含 D:\\ 盘位 glob）
      3. 当前进程解释器 sys.executable（GUI 用什么环境启动就复用哪个）
      4. conda base
    返回第一个存在的路径；全部找不到返回 None
    """
    candidates = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(conda_exe)),
                                       "envs", "yolo", "python.exe"))
    for base in (os.path.expanduser("~"), r"C:\ProgramData", "C:\\"):
        for name in ("anaconda3", "miniconda3", "Anaconda3", "Miniconda3"):
            candidates.append(os.path.join(base, name, "envs", "yolo", "python.exe"))
    for drive in ("D:\\", "E:\\"):
        candidates += glob.glob(os.path.join(drive, "*", "envs", "yolo", "python.exe"))

    if sys.executable and os.path.isfile(sys.executable):
        exe = sys.executable
        if exe.lower().endswith("pythonw.exe"):   # GUI 常以 pythonw 启动，子进程统一用 python.exe
            exe = exe[:-len("pythonw.exe")] + "python.exe"
        candidates.append(exe)

    if conda_exe and os.path.isfile(conda_exe):
        candidates.append(conda_exe)
    for base in (os.path.expanduser("~"), r"C:\ProgramData", "C:\\"):
        for name in ("anaconda3", "miniconda3", "Anaconda3", "Miniconda3"):
            candidates.append(os.path.join(base, name, "python.exe"))

    seen = set()
    for p in candidates:
        p = os.path.normpath(p)
        if p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------- #
# 图像读写封装（中文路径安全）
# ---------------------------------------------------------------- #

def save_frame(frame, path, quality=95):
    """cv2.imwrite 的中文路径安全版"""
    import cv2
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise JobError(f"图像编码失败: {path}")
    buf.tofile(path)


def read_image(path):
    """cv2.imread 的中文路径安全版，失败返回 None"""
    import cv2
    import numpy as np
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _open_video(path):
    """打开视频；中文路径直接打开失败时尝试 8.3 短路径重试"""
    import cv2
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap
    if any(ord(c) > 127 for c in path):
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(2048)
            if ctypes.windll.kernel32.GetShortPathNameW(path, buf, 2048):
                short = buf.value
                if short and short != path:
                    cap2 = cv2.VideoCapture(short)
                    if cap2.isOpened():
                        return cap2
        except Exception:
            pass
    return cap  # 未打开成功，交由调用方报错


# ---------------------------------------------------------------- #
# 步骤① 视频抽帧
# ---------------------------------------------------------------- #

def _thumb64(frame):
    """灰度 + 缩放到 64x64，用于相似帧去重比较"""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 64))


def run_extract(video_dir, out_dir, fps_extract, progress_every=200,
                dedup=False, dedup_threshold=90.0):
    """
    把 video_dir 下每个视频按每秒 fps_extract 帧抽样保存为 jpg 到 out_dir。
    命名规则与用户原有 extract_frames.py 保持一致: <视频名>_<5位序号>.jpg

    dedup=True 时启用相似帧去重：与上一个【保留】帧比较灰度缩略图，
    相似度=(1-平均绝对差/255)*100，≥ dedup_threshold(0~100) 视为重复丢弃。
    阈值=100 时只有完全相同才丢弃，等效不去重。
    """
    import cv2
    import numpy as np

    if dedup and not (0.0 <= dedup_threshold <= 100.0):
        raise JobError(f"去重阈值不合法: {dedup_threshold} (应为 0~100)")
    if fps_extract is None or fps_extract <= 0:
        raise JobError(f"每秒抽帧数不合法: {fps_extract}")
    if not os.path.isdir(video_dir):
        raise JobError(f"视频文件夹不存在: {video_dir}")
    videos = [f for f in os.listdir(video_dir)
              if f.lower().endswith(VIDEO_EXTS) and os.path.isfile(os.path.join(video_dir, f))]
    if not videos:
        raise JobError(f"视频文件夹里没有视频文件: {video_dir}")
    os.makedirs(out_dir, exist_ok=True)

    total_saved = 0
    processed_all = 0
    total_dedup = 0
    t0 = time.time()

    for vi, vname in enumerate(videos, 1):
        vpath = os.path.join(video_dir, vname)
        stem = os.path.splitext(vname)[0]
        cap = _open_video(vpath)
        if not cap.isOpened():
            log(f"[警告] 无法打开视频，已跳过: {vname}")
            continue

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 0:
            src_fps = 25.0  # 元数据缺失时兜底
        frame_interval = int(src_fps / fps_extract)
        if frame_interval < 1:
            frame_interval = 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        log(f"▶ [{vi}/{len(videos)}] {vname} (原帧率{src_fps:.1f}, 每{frame_interval}帧取1张"
            + (f", 去重阈值{dedup_threshold:.0f}%" if dedup else "") + ")")
        saved_count = 0
        last_thumb = None
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % frame_interval == 0:
                if dedup and last_thumb is not None:
                    thumb = _thumb64(frame)
                    diff = float(np.mean(np.abs(
                        thumb.astype(np.int16) - last_thumb.astype(np.int16)))) / 255.0
                    if (1.0 - diff) * 100.0 >= dedup_threshold:
                        total_dedup += 1
                        frame_index += 1
                        processed_all += 1
                        continue
                    last_thumb = thumb
                elif dedup:
                    last_thumb = _thumb64(frame)
                img_name = f"{stem}_{saved_count + 1:05d}.jpg"
                save_frame(frame, os.path.join(out_dir, img_name))
                saved_count += 1
                total_saved += 1
            frame_index += 1
            processed_all += 1
            if progress_every and processed_all % progress_every == 0:
                log(f"PROGRESS {processed_all}/{max(total_frames, processed_all)}")
        cap.release()
        log(f"  ✔ {vname}: 共保存 {saved_count} 张")

    secs = time.time() - t0
    dedup_txt = f"，去重跳过 {total_dedup} 帧" if dedup else ""
    log(f"＝ 抽帧完成: {len(videos)} 个视频 → {total_saved} 张图片{dedup_txt} ({secs:.1f}s)")
    log(f"  输出目录: {out_dir}")


# ---------------------------------------------------------------- #
# 步骤② 抽取20%移动到 dataset/
# ---------------------------------------------------------------- #

def run_move20(images_all_dir, dataset_root, sample_ratio=0.2,
               inner_train_ratio=0.7, class_names=("gangzhu",)):
    """
    从 images_all_dir 随机抽取 sample_ratio 比例的图片【移动】到:
        dataset_root/images/train  (占抽中部分的 inner_train_ratio)
        dataset_root/images/val    (其余)
    并创建 labels/{train,val} 与 classes.txt。
    返回统计 dict。
    """
    imgs = list_images(images_all_dir)
    if not imgs:
        raise JobError(f"来源目录没有图片: {images_all_dir}")

    ds_images = os.path.join(dataset_root, "images")
    ds_labels = os.path.join(dataset_root, "labels")
    train_img_dir = os.path.join(ds_images, "train")
    val_img_dir = os.path.join(ds_images, "val")
    train_lbl_dir = os.path.join(ds_labels, "train")
    val_lbl_dir = os.path.join(ds_labels, "val")
    for d in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
        os.makedirs(d, exist_ok=True)

    # 防呆：dataset 里已有图说明跑过一次，重复执行会造成两批混合，拒绝
    already = len(list_images(train_img_dir)) + len(list_images(val_img_dir))
    if already:
        raise JobError(
            f"dataset/images 下已有 {already} 张图片，疑似已执行过抽取。\n"
            f"如需重来请先清空 {ds_images} 与 {ds_labels}")

    # 写 classes.txt；若已有标注且类名变了则拒绝（防止索引错位污染标签）
    classes_txt = os.path.join(dataset_root, "classes.txt")
    new_cls = [c.strip() for c in class_names if c.strip()]
    if not new_cls:
        raise JobError("类别名为空，至少填写一个类别")
    old_cls = read_classes(classes_txt)
    has_labels = any(
        f.lower().endswith(".txt")
        for d in (train_lbl_dir, val_lbl_dir) if os.path.isdir(d)
        for f in os.listdir(d)
    )
    if has_labels and old_cls and old_cls != new_cls:
        raise JobError("检测到已有标注文件且类别名发生修改！\n"
                       "旧标注使用旧的类别索引，直接改类名会导致标签错位。\n"
                       "如确需修改，请先处理已有标注文件。")
    with open(classes_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(new_cls) + "\n")

    random.shuffle(imgs)
    sample_size = max(1, int(len(imgs) * sample_ratio))
    sampled = imgs[:sample_size]
    train_count = max(1, int(len(sampled) * inner_train_ratio))
    train_files = sampled[:train_count]
    val_files = sampled[train_count:]

    moved_train, moved_val, skipped = 0, 0, []
    for name in train_files:
        dst = os.path.join(train_img_dir, name)
        if os.path.exists(dst):
            skipped.append(name)
            continue
        shutil.move(os.path.join(images_all_dir, name), dst)
        moved_train += 1
    for name in val_files:
        dst = os.path.join(val_img_dir, name)
        if os.path.exists(dst):
            skipped.append(name)
            continue
        shutil.move(os.path.join(images_all_dir, name), dst)
        moved_val += 1

    log("＝ 抽取完成:")
    log(f"  来源总数       : {len(imgs)}")
    log(f"  抽中           : {sample_size} 张 ({sample_ratio:.0%})")
    log(f"    → train     : {moved_train} 张")
    log(f"    → val       : {moved_val} 张")
    log(f"  剩余待预测     : {len(list_images(images_all_dir))} 张 (留在原地)")
    if skipped:
        log(f"  [警告] 目标重名跳过 {len(skipped)} 张: {skipped[:5]}")
    log(f"  类别({len(new_cls)}): {', '.join(new_cls)} → {classes_txt}")
    return {"total": len(imgs), "sampled": sample_size,
            "train": moved_train, "val": moved_val}


# ---------------------------------------------------------------- #
# 图片↔标签配对
# ---------------------------------------------------------------- #

def pair_images_labels(images_dir, labels_dir):
    """按同名basename配对图片与.txt标签。返回 (配对列表[(img,lbl)], 缺失标签图片名列表)"""
    pairs, missing = [], []
    for img_name in list_images(images_dir):
        stem = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(labels_dir, stem + ".txt")
        if os.path.isfile(lbl_path):
            pairs.append((os.path.join(images_dir, img_name), lbl_path))
        else:
            missing.append(img_name)
    return pairs, missing


# ---------------------------------------------------------------- #
# 步骤⑥ 最终划分 (复制, 70/20/10)
# ---------------------------------------------------------------- #

def run_split(images_dir, labels_dir, out_root,
              r_train=0.7, r_val=0.2, empty_bg=False):
    """
    将 images_dir + labels_dir 的配对样本【复制】划分为:
        out_root/{images,labels}/{train,val,test}
    比例 r_train / r_val / 其余为 test。原件不动。

    empty_bg=True 时：没有标签的图片（预测没检出目标）生成 0 字节空 txt，
    YOLO 训练视空标签为纯背景负样本，可提升精确率。
    注意风险：若图中实际有目标但第一轮漏检，会被教成背景。
    """
    r_test = 1.0 - r_train - r_val
    if r_test < -1e-9:
        raise JobError(f"比例之和超过 100%: train={r_train:g}, val={r_val:g}")
    if not os.path.isdir(images_dir):
        raise JobError(f"图片目录不存在: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise JobError(f"标签目录不存在: {labels_dir}")

    out_images = os.path.join(out_root, "images")
    out_labels = os.path.join(out_root, "labels")
    subdirs = ("train", "val", "test")
    made_any = False
    for split in subdirs:
        for base in (out_images, out_labels):
            d = os.path.join(base, split)
            if not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            elif os.listdir(d):
                made_any = True
    if made_any:
        raise JobError(f"输出目录已有内容，请先删除或改名后再划分:\n  {out_root}")

    if empty_bg:
        # 无检出图片当背景样本：在标签目录生成空 txt，随图一起复制进 final_dataset
        os.makedirs(labels_dir, exist_ok=True)
        made = 0
        for img_name in list_images(images_dir):
            stem = os.path.splitext(img_name)[0]
            lbl = os.path.join(labels_dir, stem + ".txt")
            if not os.path.isfile(lbl):
                with open(lbl, "w", encoding="utf-8"):
                    pass
                made += 1
        if made:
            log(f"[背景图] 为 {made} 张无检出图片生成空标签（当背景样本纳入）")

    pairs, missing = pair_images_labels(images_dir, labels_dir)
    if not pairs:
        raise JobError("没有找到任何『图片+标签』配对，请确认标签 txt 与图片同名")

    random.shuffle(pairs)
    total = len(pairs)
    train_count = int(total * r_train)
    val_count = int(total * r_val)
    buckets = [
        ("train", pairs[:train_count]),
        ("val",   pairs[train_count:train_count + val_count]),
        ("test",  pairs[train_count + val_count:]),   # 余数全给 test，避免取整丢失
    ]

    copied = {"train": 0, "val": 0, "test": 0}
    skipped_dup = []
    for split_name, plist in buckets:
        for img_src, lbl_src in plist:
            dst_img = os.path.join(out_images, split_name, os.path.basename(img_src))
            dst_lbl = os.path.join(out_labels, split_name, os.path.basename(lbl_src))
            if os.path.exists(dst_img) or os.path.exists(dst_lbl):
                skipped_dup.append(os.path.basename(img_src))
                continue
            shutil.copy2(img_src, dst_img)
            shutil.copy2(lbl_src, dst_lbl)
            copied[split_name] += 1

    log("＝ 最终划分完成 (复制模式, 原件未动):")
    log(f"  有效配对: {total} 对")
    for split_name in ("train", "val", "test"):
        log(f"    {split_name:<5}: {copied[split_name]} 张")
    if missing:
        log(f"  [注意] {len(missing)} 张图片没有标签被跳过 (前10条):")
        for name in missing[:10]:
            log(f"    - {name}")
    if skipped_dup:
        log(f"  [警告] 目标重名跳过 {len(skipped_dup)} 张")
    log(f"  输出目录: {out_root}")
    return {"total": total, **copied, "missing": len(missing)}


# ---------------------------------------------------------------- #
# dataset.yaml 生成
# ---------------------------------------------------------------- #

def build_yaml(dataset_root, names, out_path):
    """生成 ultralytics dataset.yaml（正斜杠绝对路径），返回 out_path"""
    if not names:
        raise JobError("类别名为空，无法生成 yaml（请先完成步骤②）")
    p = os.path.abspath(dataset_root).replace("\\", "/").rstrip("/")
    lines = [
        "# 本文件由 YOLO数据集工具箱 自动生成",
        f'path: "{p}"',
        f'train: "{p}/images/train"',
        f'val: "{p}/images/val"',
    ]
    if os.path.isdir(os.path.join(p, "images", "test")):
        lines.append(f'test: "{p}/images/test"')
    lines.append("names:")
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"已生成数据集配置: {out_path}")
    return out_path


# ---------------------------------------------------------------- #
# 步骤④⑦ 训练 / 步骤⑤ 预测
# ---------------------------------------------------------------- #

def run_train(data, model, project_root, epochs=300, imgsz=224, batch=32,
              patience=50, copy_paste=0.3, workers=2, cache_ram=False):
    """调用 ultralytics 训练一个模型，返回实际保存目录"""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if not os.path.isfile(data):
        raise JobError(f"数据集 yaml 不存在: {data}")
    from ultralytics import YOLO

    kwargs = dict(data=os.path.abspath(data), epochs=epochs, imgsz=imgsz,
                  batch=batch, patience=patience, copy_paste=copy_paste,
                  workers=workers)
    if cache_ram:
        kwargs["cache"] = "ram"

    project = os.path.join(project_root, "result")
    name = os.path.splitext(os.path.basename(model))[0] or "model"
    log(f"＝ 开始训练: {model}  (epochs={epochs}, imgsz={imgsz}, batch={batch})")
    y = YOLO(model)
    y.train(project=project, name=name, **kwargs)
    save_dir = getattr(getattr(y, "trainer", None), "save_dir", "") or \
               os.path.join(project, name)
    log(f"＝ 训练完成: {save_dir}")
    log(f"SAVE_DIR: {save_dir}")
    return save_dir


def run_predict(weights, source_dir, conf, project, name):
    """对整个文件夹预测并保存框坐标txt，返回实际保存目录"""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if not os.path.isfile(weights):
        raise JobError(f"权重文件不存在: {weights}")
    if not os.path.isdir(source_dir):
        raise JobError(f"预测图片目录不存在: {source_dir}")
    if not list_images(source_dir):
        raise JobError(f"预测图片目录里没有图片: {source_dir}")
    from ultralytics import YOLO

    log(f"＝ 开始预测: {source_dir}  (conf={conf})")
    y = YOLO(weights)
    y.predict(source=source_dir, save=True, show=False, save_txt=True,
              conf=conf, project=project, name=name, exist_ok=True)
    save_dir = os.path.join(project, name)
    log(f"＝ 预测完成: {save_dir}")
    log(f"SAVE_DIR: {save_dir}")
    return save_dir


def post_predict_copy(predict_labels_dir, rest_labels_dir):
    """把预测产生的 labels/*.txt 拷贝为与剩余图片同级的 rest_labels 目录，返回拷贝数"""
    if not os.path.isdir(predict_labels_dir):
        return 0
    os.makedirs(rest_labels_dir, exist_ok=True)
    n = 0
    for txt in glob.glob(os.path.join(predict_labels_dir, "*.txt")):
        shutil.copy2(txt, os.path.join(rest_labels_dir, os.path.basename(txt)))
        n += 1
    return n


def collect_best_pt(search_root):
    """递归收集项目根下所有 best.pt，按修改时间降序"""
    hits = []
    if not search_root or not os.path.isdir(search_root):
        return hits
    for dirpath, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".trash")]
        if "best.pt" in filenames:
            p = os.path.join(dirpath, "best.pt")
            try:
                hits.append((os.path.getmtime(p), p))
            except OSError:
                pass
    hits.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in hits]


def read_results_summary(save_dir):
    """
    读取训练 run 的 results.csv 末行指标，返回 dict（列名→数值）。
    文件不存在或格式异常返回 None。用于多模型对比汇总。
    """
    csv_path = os.path.join(save_dir, "results.csv")
    if not os.path.isfile(csv_path):
        return None
    try:
        with open(csv_path, encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        if len(lines) < 2:
            return None
        header = [h.strip() for h in lines[0].split(",")]
        last = lines[-1].split(",")
        out = {}
        for h, v in zip(header, last):
            try:
                out[h] = float(v.strip())
            except ValueError:
                out[h] = v.strip()
        return out
    except Exception:
        return None


# ---------------------------------------------------------------- #
# 命令行入口（子进程 job 分发）
# ---------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="YOLO数据集工具箱 核心作业进程")
    ap.add_argument("--job", required=True,
                    choices=["extract", "train", "predict"])

    g1 = ap.add_argument_group("extract")
    g1.add_argument("--video_dir")
    g1.add_argument("--out_dir")
    g1.add_argument("--fps", type=float, default=5.0)
    g1.add_argument("--dedup", type=int, default=0,
                    help="1=启用相似帧去重")
    g1.add_argument("--dedup_threshold", type=float, default=90.0,
                    help="去重相似度阈值 0~100 (默认90)")

    g2 = ap.add_argument_group("train")
    g2.add_argument("--data")
    g2.add_argument("--model", default="yolov8n.pt")
    g2.add_argument("--project_root")
    g2.add_argument("--epochs", type=int, default=300)
    g2.add_argument("--imgsz", type=int, default=224)
    g2.add_argument("--batch", type=int, default=32)
    g2.add_argument("--patience", type=int, default=50)
    g2.add_argument("--copy_paste", type=float, default=0.3)
    g2.add_argument("--workers", type=int, default=2)
    g2.add_argument("--cache_ram", type=int, default=0)

    g3 = ap.add_argument_group("predict")
    g3.add_argument("--weights")
    g3.add_argument("--source")
    g3.add_argument("--conf", type=float, default=0.25)
    g3.add_argument("--project")
    g3.add_argument("--name", default="predict1")

    args = ap.parse_args()

    try:
        if args.job == "extract":
            if not args.video_dir or not args.out_dir:
                raise JobError("extract 需要 --video_dir 和 --out_dir")
            run_extract(args.video_dir, args.out_dir, args.fps,
                        dedup=bool(args.dedup),
                        dedup_threshold=args.dedup_threshold)
        elif args.job == "train":
            if not args.data or not args.project_root:
                raise JobError("train 需要 --data 和 --project_root")
            run_train(args.data, args.model, args.project_root,
                      epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                      patience=args.patience, copy_paste=args.copy_paste,
                      workers=args.workers, cache_ram=bool(args.cache_ram))
        elif args.job == "predict":
            if not args.weights or not args.source or not args.project:
                raise JobError("predict 需要 --weights --source --project")
            run_predict(args.weights, args.source, args.conf,
                        args.project, args.name)
    except JobError as e:
        log(f"JOB_ERROR: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        log("已被中断")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
