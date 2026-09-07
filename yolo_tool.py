#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO数据集工具箱 —— GUI 主程序
================================
版本: 3.1 (2026-08-26) —— 高级参数悬停介绍
运行方式: 用带 ultralytics 的 python 环境执行（推荐双击 启动工具.bat）

七个页签对应完整工作流:
  ①视频抽帧 → ②抽取20% → ③启动标注 → ④第一次训练
  → ⑤预测剩余80% → ⑥微调+最终划分 → ⑦第二次训练
"""

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import tool_core as core

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "config.json")

DEFAULT_CONFIG = {
    "version": 2,
    # 环境路径留空 = 首次启动自动探测（find_python_exe / find_labelimg_exe）
    "python_exe": "",
    "labelimg_exe": "",
    "project_root": "",
    "recent_projects": [],
    "classes_text": "gangzhu",
    "tab1": {"video_dir": "", "fps": 5.0,
             "dedup_on": False, "dedup_threshold": 90},
    "tab2": {"sample_ratio": 0.2, "inner_train_ratio": 0.7},
    "tab4": {"models": ["yolov8n.pt"], "epochs": 300, "imgsz": 224, "batch": 32,
             "patience": 50, "copy_paste": 0.3, "workers": 2, "cache_ram": False,
             "adv_show": False,
             "adv": {"lr0": "", "lrf": "", "weight_decay": "", "mosaic": "",
                     "fliplr": "", "degrees": "", "cos_lr": True,
                     "device": "", "seed": ""}},
    "tab5": {"conf": 0.25},
    "tab6": {"r_train": 0.7, "r_val": 0.2, "empty_bg": False},
    "tab7": {"models": ["yolov8n.pt"], "epochs": 300, "imgsz": 224, "batch": 32,
             "patience": 50, "copy_paste": 0.3, "workers": 2, "cache_ram": False,
             "adv_show": False,
             "adv": {"lr0": "", "lrf": "", "weight_decay": "", "mosaic": "",
                     "fliplr": "", "degrees": "", "cos_lr": True,
                     "device": "", "seed": ""}},
}

# 预设模型下拉值（多选）
PRESET_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt"]

# 高级参数详细说明（v3.1：悬停 ❓ 显示）
ADV_DESCRIPTIONS = {
    "lr0": "【初始学习率】\n"
           "每次更新权重的步长，最关键的调参项之一。\n"
           "· 太大(>0.05)：loss 震荡不收敛，甚至发散\n"
           "· 太小(<0.0001)：收敛极慢，且易陷入局部最优\n"
           "· 建议：用预训练权重微调常用 0.01；从头训练或小数据集\n"
           "  可降到 0.001~0.005；数据集很小(几十张)时 0.0005 更稳",
    "lrf": "【终值学习率系数】\n"
           "训练结束时学习率降到多少，最终 lr = lr0 × lrf。\n"
           "· 默认 0.01 = 结束时降为初始的 1/100\n"
           "· 越大说明结束时的学习率越高，后期调参幅度大\n"
           "· 一般保持默认即可，通常无需修改",
    "weight_decay": "【权重衰减 / L2 正则化】\n"
                    "惩罚过大权重，抑制过拟合。\n"
                    "· 默认 0.0005 适合大多数场景\n"
                    "· 数据量少、容易过拟合时调大(0.001~0.005)\n"
                    "· 欠拟合(训练/验证损失都高)时可调小甚至归零",
    "mosaic": "【马赛克增强】\n"
              "把 4 张图拼成 1 张训练，大幅增加样本多样性，\n"
              "提升小目标和遮挡场景的鲁棒性。\n"
              "· 默认 1.0（每个 batch 都做）\n"
              "· 数据集本身很小(≤几百张)或目标密集难分时，\n"
              "  建议降到 0.5 或 0，否则合成图太乱影响学习\n"
              "· ultralytics 会在最后 10 个 epoch 自动关闭它",
    "fliplr": "【水平翻转增强】\n"
              "随机水平镜像的概率（0~1）。\n"
              "· 默认 0.5 = 一半图片翻转，通用增强手段\n"
              "· 若目标有左右语义（如车牌文字、方向箭头、\n"
              "  左右不对称部件），必须设为 0 防止学错方向",
    "degrees": "【随机旋转增强】\n"
               "图片随机旋转的角度范围（度）。\n"
               "· 默认 0 = 不旋转\n"
               "· 目标姿态多变时设 5~30，帮助模型学旋转不变性\n"
               "· 注意：旋转会带来框的噪声，数值不宜过大；\n"
               "  若检测物有明确上下(如文字)则保持 0",
    "cos_lr": "【余弦退火学习率调度】\n"
              "学习率按余弦曲线从初始值平滑下降到 lrf。\n"
              "· 相比直线下降，后期衰减更缓慢平缓，收敛通常更稳\n"
              "· ultralytics 默认开启，一般建议保持勾选\n"
              "· 个别场景(短训练/实验对比)可取消改为固定线性下降",
    "device": "【训练设备】\n"
              "指定用哪块 GPU 或 CPU 训练。\n"
              "· 0 = 第一块显卡；0,1 = 两张卡并行(需显存够)\n"
              "· cpu = 用 CPU 训练（非常慢，不推荐）\n"
              "· 留空 = 自动选择（有 GPU 用 GPU）",
    "seed": "【随机种子】\n"
            "固定随机种子后，两次训练的数据打乱/增强顺序一致，\n"
            "结果可复现，便于做对照实验。\n"
            "· 默认 0；改一个不同数字 = 换一种随机轨迹\n"
            "· 做「同一参数跑两次看波动」或论文复现时有用",
}


# ================================================================ #
# 配置持久化
# ================================================================ #

class Config:
    def __init__(self):
        self.data = json.loads(json.dumps(DEFAULT_CONFIG))  # 深拷贝默认值
        self.load()

    def load(self):
        if not os.path.isfile(CONFIG_PATH):
            self.auto_detect()   # 全新机器：默认值空白，自动探测环境
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                disk = json.load(f)
            for k, v in disk.items():
                if k == "version":
                    continue        # 版本号始终以程序内置为准
                if isinstance(v, dict) and isinstance(self.data.get(k), dict):
                    self.data[k].update(v)   # tab 子字典合并，兼容新增字段
                else:
                    self.data[k] = v
            self._migrate_v1()
            self.auto_detect()   # 配置里缺的环境路径用自动探测补齐
        except Exception:
            # 配置损坏：备份后用默认值
            try:
                shutil.copy2(CONFIG_PATH, CONFIG_PATH + ".bak")
            except Exception:
                pass
            self.auto_detect()

    def _migrate_v1(self):
        """v1 → v2: 训练页签单值 model 字段 → models 列表"""
        for key in ("tab4", "tab7"):
            t = self.data.get(key, {})
            if isinstance(t, dict) and "model" in t:
                m = (t.get("model") or "").strip()
                # 磁盘旧值优先（若为空则保留默认列表）
                if m:
                    t["models"] = [m]
                t.pop("model", None)

    def auto_detect(self):
        """python_exe / labelimg_exe 为空时自动探测，让新机器 clone 即用"""
        if not self.data.get("python_exe"):
            found = core.find_python_exe()
            self.data["python_exe"] = found or ""
        if not self.data.get("labelimg_exe"):
            found = core.find_labelimg_exe()
            self.data["labelimg_exe"] = found or ""

    def save(self):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def setdefault(self, key, default=None):
        return self.data.setdefault(key, default)

    def __setitem__(self, key, value):
        self.data[key] = value


# ================================================================ #
# 悬停提示 (v3.1: 高级参数说明用, 纯 tkinter 实现)
# ================================================================ #

class Tooltip:
    """鼠标悬停 0.5 秒后显示说明气泡，离开即隐藏"""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_win = None
        self.after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e=None):
        self._cancel()
        self.after_id = self.widget.after(500, self._show)

    def _show(self):
        if self.tip_win is not None:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip_win = tk.Toplevel(self.widget)
        self.tip_win.wm_overrideredirect(True)
        self.tip_win.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(self.tip_win, text=self.text, justify=tk.LEFT,
                       background="#ffffe6", foreground="#222",
                       relief=tk.SOLID, borderwidth=1,
                       font=("Microsoft YaHei UI", 9),
                       wraplength=420, padx=10, pady=8)
        lbl.pack()

    def _hide(self, _e=None):
        self._cancel()
        if self.tip_win is not None:
            try:
                self.tip_win.destroy()
            except Exception:
                pass
            self.tip_win = None

    def _cancel(self):
        if self.after_id is not None:
            try:
                self.widget.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None


# ================================================================ #
# 子进程任务管理器（同一时刻至多一个重活）
# ================================================================ #

class JobRunner:
    """
    Popen + 读线程 + 队列轮询。GUI 只在主线程碰 tk 对象。
    支持任务队列：连续 start() 会排队逐个执行；stop() 终止当前并清空队列。
    """

    def __init__(self, root, on_line=None, on_exit=None):
        self.root = root
        self.q = queue.Queue()
        self.proc = None
        self.on_line = on_line or (lambda line: None)
        self.on_exit = on_exit or (lambda code: None)
        self.stopped_by_user = False
        self.pending = []          # [(cmd_list, desc, on_done)] 待执行队列
        self.desc = ""             # 当前任务描述
        self.start_time = 0.0      # 当前任务启动时间（用于统计用时）
        self.last_elapsed = 0.0    # 上一个任务实际耗时（秒）
        self._on_done = None

    @property
    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def queue_len(self):
        """当前执行中 + 排队中的任务数"""
        return (1 if self.is_running else 0) + len(self.pending)

    def start(self, cmd_list, desc="", on_done=None):
        """入队并启动（若当前空闲则立即执行第一个）"""
        self.pending.append((cmd_list, desc, on_done))
        if not self.is_running:
            self._launch_next()

    def _launch_next(self):
        if not self.pending:
            return
        cmd_list, desc, on_done = self.pending.pop(0)
        self.desc = desc
        self._on_done = on_done
        self.stopped_by_user = False
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=env, cwd=APP_DIR, creationflags=creationflags)
        self.start_time = time.time()

        def reader():
            try:
                for line in self.proc.stdout:
                    self.q.put(line.rstrip("\r\n"))
            finally:
                self.q.put(None)

        threading.Thread(target=reader, daemon=True).start()
        self.root.after(100, self._poll)

    def _poll(self):
        alive = True
        while True:
            try:
                line = self.q.get_nowait()
            except queue.Empty:
                break
            if line is None:                      # 结束哨兵
                alive = False
                continue
            if "%|" in line:                      # tqdm 进度条只保留末帧
                continue
            self.on_line(line)
        if alive and self.is_running:
            self.root.after(100, self._poll)
        elif not alive:
            # stdout 关闭 ≠ 进程退出（CUDA 清理等可能延后），
            # 必须真正 wait() 拿准确退出码，否则会误报 -1 失败
            proc, self.proc = self.proc, None
            if proc is not None:
                try:
                    code = proc.wait(timeout=10)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    code = -1
            else:
                code = -1
            self.last_elapsed = time.time() - self.start_time
            self.on_exit(code if code is not None else -1)
            # 队列续跑（用户主动停止则清空不续）
            if not self.stopped_by_user and self.pending:
                self._launch_next()

    def stop(self):
        """终止当前任务并清空队列（多模型排队时一键全停）"""
        if not self.proc:
            self.pending.clear()
            return
        self.stopped_by_user = True
        self.pending.clear()

        def killer():
            try:
                self.proc.terminate()
            except Exception:
                return
            try:
                self.proc.wait(3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

        threading.Thread(target=killer, daemon=True).start()


# ================================================================ #
# GUI 主应用
# ================================================================ #

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO 数据集工具箱")
        self.geometry("900x720")
        self.minsize(860, 660)

        self.cfg = Config()
        self.runner = JobRunner(self, on_line=self._on_job_line,
                                on_exit=self._on_job_exit)
        self.start_buttons = []          # 所有需要互斥禁用的按钮
        self.stop_buttons = {}           # name -> button
        self.train_tabs_meta = {}        # tab4/tab7 -> yaml kind
        self._adv_frames = {}            # tab4/tab7 -> (高级选项frame, 展开状态var)
        self._last_save_dir = ""
        self._on_done_hook = None
        self._train_batch = None         # 多模型对比: [{model, save_dir, elapsed}]
        self._batch_running = False

        self._build_ui()
        self._load_config_to_widgets()
        self.refresh_yaml_labels()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.log("欢迎使用 YOLO 数据集工具箱。请先在顶部选择项目根目录。")
        self.log("流程建议按页签顺序: 抽帧→抽取20%→标注→训练→预测→微调划分→二次训练")
        self._report_env()

    def _report_env(self):
        """启动时展示环境探测结果，缺失时给出明确指引"""
        py = self.cfg["python_exe"]
        li = self.cfg["labelimg_exe"]
        if py and os.path.isfile(py):
            self.log(f"Python 解释器: {py}")
        else:
            self.log("[警告] 未找到可用的 Python 解释器，训练/预测将不可用。\n"
                     "       请修改 config.json 的 python_exe，"
                     "指向装有 ultralytics 的 python.exe", error=True)
        if li and os.path.isfile(li):
            self.log(f"labelImg: {li}")
        else:
            self.log("[警告] 未找到 labelImg，标注功能不可用。\n"
                     "       可在页签③点『自动检测路径』或手动浏览选择", error=True)

    # ---------------- 基础设施 ---------------- #

    def log(self, msg, error=False):
        self.logbox.configure(state=tk.NORMAL)
        self.logbox.insert(tk.END, str(msg) + "\n")
        if error:
            self.logbox.tag_add("err", "end-2l", "end-1l")
            self.logbox.tag_configure("err", foreground="#c0392b")
        self.logbox.see(tk.END)
        self.logbox.configure(state=tk.DISABLED)

    def set_busy(self, busy, stop_name=None):
        for b in self.start_buttons:
            b.configure(state=tk.DISABLED if busy else tk.NORMAL)
        for name, b in self.stop_buttons.items():
            b.configure(state=(tk.NORMAL if (busy and name == stop_name)
                               else tk.DISABLED))

    def save_config_from_widgets(self):
        c = self.cfg
        c["project_root"] = self.var_root.get().strip()
        c["classes_text"] = self.txt_classes.get("1.0", tk.END).strip()
        c["labelimg_exe"] = self.var_labelimg.get().strip()
        c["tab1"].update({"video_dir": self.var_video_dir.get().strip(),
                          "fps": float(self.var_fps.get()),
                          "dedup_on": bool(self.var_dedup.get()),
                          "dedup_threshold": float(self.var_dedup_th.get())})
        c["tab2"].update({"sample_ratio": float(self.var_sample.get()),
                          "inner_train_ratio": float(self.var_inner.get())})
        for key in ("tab4", "tab7"):
            v = getattr(self, f"{key}_vars", None)
            if v is None:
                continue
            adv, _ = self._collect_adv(v)
            c[key].update({
                "models": self._collect_models(v),
                "epochs": int(v["epochs"].get()),
                "imgsz": int(v["imgsz"].get()),
                "batch": int(v["batch"].get()),
                "patience": int(v["patience"].get()),
                "copy_paste": float(v["copy_paste"].get()),
                "workers": int(v["workers"].get()),
                "cache_ram": bool(v["cache"].get()),
                "adv_show": bool(v["adv_show"].get()),
                "adv": adv})
        c["tab5"].update({"conf": float(self.var_conf.get())})
        c["tab6"].update({"r_train": float(self.var_rtrain.get()),
                          "r_val": float(self.var_rval.get()),
                          "empty_bg": bool(self.var_empty_bg.get())})
        # 最近项目历史（去重置顶，最多5个）
        root = c["project_root"]
        recents = c.setdefault("recent_projects", [])
        if root and root in recents:
            recents.remove(root)
        if root:
            recents.insert(0, root)
        c["recent_projects"] = recents[:5]
        self.cb_root["values"] = c["recent_projects"]
        self.cfg.save()

    def _load_config_to_widgets(self):
        c = self.cfg
        self.var_root.set(c["project_root"])
        self.cb_root["values"] = c.get("recent_projects", [])
        self.txt_classes.delete("1.0", tk.END)
        self.txt_classes.insert(tk.END, c["classes_text"])
        self.var_labelimg.set(c["labelimg_exe"])
        t1, t2 = c["tab1"], c["tab2"]
        self.var_video_dir.set(t1["video_dir"])
        self.var_fps.set(t1["fps"])
        self.var_dedup.set(bool(t1.get("dedup_on", False)))
        self.var_dedup_th.set(t1.get("dedup_threshold", 90))
        self.var_sample.set(t2["sample_ratio"])
        self.var_inner.set(t2["inner_train_ratio"])
        for key in ("tab4", "tab7"):
            t = c.get(key, {})
            v = getattr(self, f"{key}_vars")
            saved = t.get("models", []) or []
            for m, var in v["model_cbs"]:
                var.set(m in saved)
            v["custom"].set(",".join(x for x in saved
                                     if x not in PRESET_MODELS))
            for k in ("epochs", "imgsz", "batch", "patience",
                      "copy_paste", "workers"):
                v[k].set(t.get(k, DEFAULT_CONFIG[key][k]))
            v["cache"].set(bool(t.get("cache_ram", False)))
            # 高级选项（旧配置无 adv 字段时用默认空值）
            adv_cfg = t.get("adv", {}) or {}
            for k, var in v["adv"].items():
                var.set(adv_cfg.get(k, ""))
            v["adv_cos"].set(bool(adv_cfg.get("cos_lr", True)))
            v["adv_show"].set(bool(t.get("adv_show", False)))
            self._toggle_adv(key, v["adv_show"].get())
        self.var_conf.set(c["tab5"]["conf"])
        t6 = c["tab6"]
        self.var_rtrain.set(t6["r_train"])
        self.var_rval.set(t6["r_val"])
        self.var_empty_bg.set(bool(t6.get("empty_bg", False)))
        self._update_preview2()
        self._update_preview6()

    # 路径派生辅助
    def p_root(self):
        return self.var_root.get().strip()

    def require_root(self):
        root = self.p_root()
        if not root:
            messagebox.showerror("提示", "请先在顶部填写项目根目录")
            return None
        if not os.path.isdir(root):
            messagebox.showerror("错误", f"项目根目录不存在:\n{root}")
            return None
        return root

    def derived(self, *parts):
        root = self.require_root()
        if root is None:
            return None
        return os.path.join(root, *parts)

    def validate_python_exe(self):
        """验证 python_exe 存在且确实装有 ultralytics（结果缓存，避免反复启动子进程）"""
        py = self.cfg["python_exe"]
        if not py or not os.path.isfile(py):
            messagebox.showerror(
                "错误", f"Python 解释器不存在:\n{py}\n\n"
                        "请修改 config.json 的 python_exe，指向装有 ultralytics "
                        "的 python.exe")
            return False
        if getattr(self, "_py_checked", None) == py:
            return self._py_ok
        try:
            r = subprocess.run(
                [py, "-c", "import ultralytics"],
                capture_output=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self._py_ok = (r.returncode == 0)
        except Exception:
            self._py_ok = False
        self._py_checked = py
        if not self._py_ok:
            messagebox.showerror(
                "错误", f"该 Python 环境里没有 ultralytics:\n{py}\n\n"
                        "请修改 config.json 的 python_exe，指向正确的 conda 环境")
        return self._py_ok

    # ---------------- UI 构建 ---------------- #

    def _build_ui(self):
        top = ttk.Frame(self, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="项目根目录:").pack(side=tk.LEFT)
        self.var_root = tk.StringVar()
        self.cb_root = ttk.Combobox(top, textvariable=self.var_root, width=60)
        self.cb_root.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(top, text="浏览…", command=self.browse_root).pack(side=tk.LEFT)
        self.var_root.trace_add("write", lambda *a: self._on_root_changed())

        nb = ttk.Notebook(self)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=3)
        self.nb = nb
        self._build_tab1(nb)
        self._build_tab2(nb)
        self._build_tab3(nb)
        self._make_train_tab(nb, key="tab4", title="④ 第一次训练",
                             yaml_kind="dataset")
        self._build_tab5(nb)
        self._build_tab6(nb)
        self._make_train_tab(nb, key="tab7", title="⑦ 第二次训练",
                             yaml_kind="final_dataset")

        logf = ttk.LabelFrame(self, text="日志", padding=4)
        logf.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=6, pady=(0, 4))
        self.logbox = scrolledtext.ScrolledText(logf, height=9, state=tk.DISABLED,
                                                font=("Consolas", 9))
        self.logbox.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="空闲")
        ttk.Label(self, textvariable=self.status_var, anchor="w",
                  relief=tk.SUNKEN, padding=(6, 2)).pack(
            side=tk.BOTTOM, fill=tk.X)

    def _path_row(self, parent, label, var):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(row, text="浏览…",
                   command=lambda: self._browse_dir(var)).pack(side=tk.LEFT)

    def _browse_dir(self, var):
        d = filedialog.askdirectory(parent=self, title="选择文件夹")
        if d:
            var.set(os.path.normpath(d))

    def browse_root(self):
        d = filedialog.askdirectory(parent=self, title="选择项目根目录")
        if d:
            self.var_root.set(os.path.normpath(d))

    # ---------- 页签① 视频抽帧 ---------- #

    def _build_tab1(self, parent):
        f = ttk.Frame(parent, padding=10)
        parent.add(f, text="① 视频抽帧")
        lf = ttk.LabelFrame(f, text="输入", padding=8)
        lf.pack(fill=tk.X)
        self.var_video_dir = tk.StringVar()
        self._path_row(lf, "视频文件夹:", self.var_video_dir)
        row = ttk.Frame(lf)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="每秒抽帧数:", width=14).pack(side=tk.LEFT)
        self.var_fps = tk.DoubleVar(value=5.0)
        ttk.Spinbox(row, from_=0.5, to=30, increment=0.5, width=8,
                    textvariable=self.var_fps).pack(side=tk.LEFT)

        dedup_row = ttk.Frame(lf)
        dedup_row.pack(fill=tk.X, pady=2)
        self.var_dedup = tk.BooleanVar(value=False)
        ttk.Checkbutton(dedup_row, text="相似帧去重（相邻几乎相同的帧自动跳过）",
                        variable=self.var_dedup).pack(side=tk.LEFT)
        ttk.Label(dedup_row, text="阈值:").pack(side=tk.LEFT, padx=(10, 2))
        self.var_dedup_th = tk.DoubleVar(value=90)
        ttk.Spinbox(dedup_row, from_=50, to=100, increment=5, width=6,
                    textvariable=self.var_dedup_th).pack(side=tk.LEFT)
        ttk.Label(dedup_row, text="% (越大越严格，100=仅跳过完全相同)").pack(
            side=tk.LEFT, padx=4)

        bf = ttk.Frame(f, padding=(0, 8))
        bf.pack(fill=tk.X)
        self.btn_extract = ttk.Button(bf, text="开始抽帧 ▶",
                                      command=self.on_extract)
        self.btn_extract.pack(side=tk.LEFT)
        self.start_buttons.append(self.btn_extract)
        self.btn_stop_extract = ttk.Button(bf, text="■ 停止", state=tk.DISABLED,
                                           command=self.runner.stop)
        self.btn_stop_extract.pack(side=tk.LEFT, padx=6)
        self.stop_buttons["extract"] = self.btn_stop_extract

        tip = ("说明：支持 mp4/avi/mov/mkv/flv/wmv；输出到 <项目根>/images_all，"
               "命名 <视频名>_00001.jpg。\n剩余的图片将留在 images_all 供步骤⑤预测。")
        ttk.Label(f, text=tip, foreground="#666", wraplength=800,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=6)

    def on_extract(self):
        root = self.require_root()
        if root is None:
            return
        video_dir = self.var_video_dir.get().strip()
        if not video_dir or not os.path.isdir(video_dir):
            messagebox.showerror("错误", f"视频文件夹不存在:\n{video_dir}")
            return
        try:
            fps = float(self.var_fps.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("错误", "每秒抽帧数不合法")
            return
        out_dir = os.path.join(root, "images_all")
        self.save_config_from_widgets()
        self.status_var.set("正在抽帧…")
        self.set_busy(True, "extract")
        dedup = bool(self.var_dedup.get())
        dedup_txt = f"，去重阈值{self.var_dedup_th.get():g}%" if dedup else ""
        self.log(f"▶ 开始抽帧: {video_dir} @ {fps}帧/秒{dedup_txt} → {out_dir}")
        cmd = [self.cfg["python_exe"],
               os.path.join(APP_DIR, "tool_core.py"),
               "--job", "extract", "--video_dir", video_dir,
               "--out_dir", out_dir, "--fps", str(fps)]
        if dedup:
            cmd += ["--dedup", "1",
                    "--dedup_threshold", str(self.var_dedup_th.get())]
        self.runner.start(cmd, desc="抽帧")

    # ---------- 页签② 抽取20% ---------- #

    def _build_tab2(self, parent):
        f = ttk.Frame(parent, padding=10)
        parent.add(f, text="② 抽取20%")
        lf = ttk.LabelFrame(f, text="比例设置", padding=8)
        lf.pack(fill=tk.X)

        row1 = ttk.Frame(lf)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="抽取比例:", width=14).pack(side=tk.LEFT)
        self.var_sample = tk.DoubleVar(value=0.2)
        ttk.Spinbox(row1, from_=0.05, to=0.5, increment=0.05, width=8,
                    textvariable=self.var_sample).pack(side=tk.LEFT)
        ttk.Label(row1, text="(从 images_all 中随机抽取并移动)").pack(
            side=tk.LEFT, padx=6)

        row2 = ttk.Frame(lf)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="内部 train 占比:", width=14).pack(side=tk.LEFT)
        self.var_inner = tk.DoubleVar(value=0.7)
        ttk.Spinbox(row2, from_=0.5, to=0.95, increment=0.05, width=8,
                    textvariable=self.var_inner).pack(side=tk.LEFT)
        ttk.Label(row2, text="(其余给 val)").pack(side=tk.LEFT, padx=6)

        cf = ttk.LabelFrame(f, text="类别名 (每行一个)", padding=8)
        cf.pack(fill=tk.X, pady=8)
        self.txt_classes = tk.Text(cf, height=4)
        self.txt_classes.pack(fill=tk.X)

        bf = ttk.Frame(f, padding=(0, 4))
        bf.pack(fill=tk.X)
        self.btn_move20 = ttk.Button(bf, text="执行抽取 ▶",
                                     command=self.on_move20)
        self.btn_move20.pack(side=tk.LEFT)
        self.start_buttons.append(self.btn_move20)

        self.var_preview2 = tk.StringVar(value="（完成抽帧后自动显示预计抽取数量）")
        ttk.Label(f, textvariable=self.var_preview2, foreground="#06c",
                  wraplength=800, justify=tk.LEFT).pack(anchor=tk.W, pady=2)
        self.var_sample.trace_add("write", lambda *a: self._update_preview2())
        self.var_inner.trace_add("write", lambda *a: self._update_preview2())

        tip = ("说明：从 <项目根>/images_all 随机移动抽取比例的图片到 "
               "<项目根>/dataset/images/{train,val}，并创建 labels 文件夹与 classes.txt。\n"
               "重复执行会被拒绝（防止两批混合）；如需重来请先清空 dataset 文件夹。")
        ttk.Label(f, text=tip, foreground="#666", wraplength=800,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=6)

    def on_move20(self):
        root = self.require_root()
        if root is None:
            return
        imgs_all = os.path.join(root, "images_all")
        ds = os.path.join(root, "dataset")
        n_imgs = len(core.list_images(imgs_all)) if os.path.isdir(imgs_all) else 0
        if not n_imgs:
            messagebox.showerror("错误", f"来源目录没有图片:\n{imgs_all}\n\n"
                                         "请先完成步骤①抽帧")
            return
        sample = float(self.var_sample.get())
        inner = float(self.var_inner.get())
        classes = [ln.strip() for ln in
                   self.txt_classes.get("1.0", tk.END).splitlines() if ln.strip()]
        if not classes:
            messagebox.showerror("错误", "类别名为空，至少填写一个类别")
            return
        sampled = max(1, int(n_imgs * sample))
        train_n = max(1, int(sampled * inner))
        val_n = sampled - train_n

        if not messagebox.askyesno(
                "确认抽取",
                f"将随机【移动】 {sampled} 张图片 (共{n_imgs}张):\n"
                f"  → dataset/images/train : {train_n} 张\n"
                f"  → dataset/images/val   : {val_n} 张\n"
                f"  剩余 {n_imgs - sampled} 张留在 images_all\n\n确定执行？"):
            return
        self.save_config_from_widgets()
        self.status_var.set("正在抽取…")
        self.set_busy(True)
        self.log(f"▶ 抽取: {imgs_all} → {ds}")

        def work():
            code = 0
            core.log_hook = lambda m: self.root.after(0, self.log, m)
            try:
                core.run_move20(imgs_all, ds, sample, inner,
                                class_names=tuple(classes))
            except core.JobError as e:
                self.log(f"JOB_ERROR: {e}", error=True)
                code = 2
            except Exception:
                import traceback
                self.log(traceback.format_exc(), error=True)
                code = 1
            finally:
                core.log_hook = None
            self.root.after(0, lambda: self._finish_light(code, "抽取"))

        threading.Thread(target=work, daemon=True).start()

    def _finish_light(self, code, what):
        self.set_busy(False)
        self.status_var.set("空闲")
        if code == 0:
            self.log(f"✔ {what}完成")
        else:
            self.log(f"✘ {what}失败，详见日志", error=True)

    # ---------- 页签③ 启动标注 ---------- #

    def _build_tab3(self, parent):
        f = ttk.Frame(parent, padding=10)
        parent.add(f, text="③ 启动标注")
        lf = ttk.LabelFrame(f, text="labelImg 路径", padding=8)
        lf.pack(fill=tk.X)
        self.var_labelimg = tk.StringVar()
        self._path_row(lf, "labelImg路径:", self.var_labelimg)
        ttk.Button(lf, text="自动检测路径",
                   command=self.on_detect_labelimg).pack(anchor=tk.W, pady=2)

        bf = ttk.LabelFrame(f, text="打开标注", padding=8)
        bf.pack(fill=tk.X, pady=8)
        self.btn_lbl_train = ttk.Button(bf, text="打开 train 标注 ▶",
                                        command=lambda: self.on_labelimg("train"))
        self.btn_lbl_train.pack(side=tk.LEFT, padx=(0, 8))
        self.start_buttons.append(self.btn_lbl_train)
        self.btn_lbl_val = ttk.Button(bf, text="打开 val 标注 ▶",
                                      command=lambda: self.on_labelimg("val"))
        self.btn_lbl_val.pack(side=tk.LEFT)
        self.start_buttons.append(self.btn_lbl_val)

        tip = ("说明：labelImg 会自动加载对应图片目录和 classes.txt，标签保存在同级 labels 目录。\n"
               "标注完成后进入步骤④第一次训练。")
        ttk.Label(f, text=tip, foreground="#666", wraplength=800,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=6)

    def _resolve_labelimg(self):
        exe = self.var_labelimg.get().strip()
        found = core.find_labelimg_exe(exe or None)
        if found:
            if found != exe:
                self.var_labelimg.set(found)
                self.cfg["labelimg_exe"] = found
            return found
        d = filedialog.askopenfilename(parent=self, title="选择 labelImg.exe",
                                       filetypes=[("可执行文件", "*.exe")])
        if d:
            self.var_labelimg.set(os.path.normpath(d))
            self.cfg["labelimg_exe"] = os.path.normpath(d)
            return os.path.normpath(d)
        return None

    def on_detect_labelimg(self):
        found = self._resolve_labelimg()
        if found:
            self.cfg.save()
            self.log(f"labelImg 路径: {found}")
            messagebox.showinfo("成功", f"已定位 labelImg:\n{found}")

    def _launch_labelimg(self, img_dir, lbl_dir, classes_txt, tag):
        found = self._resolve_labelimg()
        if not found:
            return
        if not os.path.isfile(classes_txt):
            names = [ln.strip() for ln in
                     self.txt_classes.get("1.0", tk.END).splitlines() if ln.strip()]
            if not names:
                messagebox.showerror("错误", "类别名为空且找不到 classes.txt")
                return
            os.makedirs(os.path.dirname(classes_txt), exist_ok=True)
            with open(classes_txt, "w", encoding="utf-8") as fo:
                fo.write("\n".join(names) + "\n")
            self.log(f"已生成 classes.txt: {classes_txt}")
        self.save_config_from_widgets()
        try:
            subprocess.Popen([found, img_dir, classes_txt, lbl_dir],
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.log(f"▶ 已启动 labelImg ({tag}): {img_dir}")
        except Exception as e:
            messagebox.showerror("启动失败", f"无法启动 labelImg:\n{e}")

    def on_labelimg(self, split):
        root = self.require_root()
        if root is None:
            return
        img_dir = os.path.join(root, "dataset", "images", split)
        lbl_dir = os.path.join(root, "dataset", "labels", split)
        classes_txt = os.path.join(root, "dataset", "classes.txt")
        if not os.path.isdir(img_dir):
            messagebox.showerror("错误", f"图片目录不存在:\n{img_dir}\n\n"
                                         "请先完成步骤①②")
            return
        self._launch_labelimg(img_dir, lbl_dir, classes_txt, split)

    # ---------- 训练页签工厂 (④/⑦共用) ---------- #

    def _make_train_tab(self, parent, key, title, yaml_kind):
        f = ttk.Frame(parent, padding=10)
        parent.add(f, text=title)
        lf = ttk.LabelFrame(f, text="训练参数", padding=8)
        lf.pack(fill=tk.X)

        row0 = ttk.Frame(lf)
        row0.pack(fill=tk.X, pady=2)
        ttk.Label(row0, text="模型(可多选):", width=14).pack(side=tk.LEFT)
        model_cbs = []
        for m in PRESET_MODELS:
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(row0, text=m.replace(".pt", ""),
                                 variable=var)
            cb.pack(side=tk.LEFT, padx=3)
            model_cbs.append((m, var))
        row_custom = ttk.Frame(lf)
        row_custom.pack(fill=tk.X, pady=2)
        ttk.Label(row_custom, text="自定义:", width=14).pack(side=tk.LEFT)
        var_custom = tk.StringVar()
        ttk.Entry(row_custom, textvariable=var_custom, width=30).pack(
            side=tk.LEFT, padx=4)
        ttk.Label(row_custom, text="(逗号分隔，可填本地路径,如 D:/xx/best.pt)",
                  foreground="#666").pack(side=tk.LEFT)

        grid = ttk.Frame(lf)
        grid.pack(fill=tk.X, pady=4)

        def add_spin(label, var, frm, to, inc, col, row_):
            ttk.Label(grid, text=label).grid(column=col, row=row_,
                                             sticky=tk.E, padx=2, pady=2)
            ttk.Spinbox(grid, from_=frm, to=to, increment=inc, width=8,
                        textvariable=var).grid(column=col + 1, row=row_,
                                               padx=2, pady=2)

        v_epochs = tk.IntVar(value=300)
        v_imgsz = tk.IntVar(value=224)
        v_batch = tk.IntVar(value=32)
        v_patience = tk.IntVar(value=50)
        v_cp = tk.DoubleVar(value=0.3)
        v_workers = tk.IntVar(value=2)
        add_spin("epochs:", v_epochs, 1, 2000, 10, 0, 0)
        add_spin("imgsz:", v_imgsz, 32, 1536, 32, 2, 0)
        add_spin("batch:", v_batch, 1, 256, 1, 4, 0)
        add_spin("patience:", v_patience, 0, 1000, 5, 0, 1)
        add_spin("copy_paste:", v_cp, 0.0, 1.0, 0.05, 2, 1)
        add_spin("workers:", v_workers, 0, 16, 1, 4, 1)

        cache_row = ttk.Frame(lf)
        cache_row.pack(fill=tk.X, pady=2)
        v_cache = tk.BooleanVar(value=False)
        ttk.Checkbutton(cache_row, text="启用 cache=ram（更快，但数据集大时可能内存不足）",
                        variable=v_cache).pack(side=tk.LEFT)

        # —— 高级选项（默认收起，展开可调学习率/增强/设备等，留空=用默认）——
        # holder 固定占位，展开/收起只动内部，不影响页签其他区域布局
        adv_holder = ttk.Frame(lf)
        adv_holder.pack(fill=tk.X, pady=(4, 0))
        var_adv_show = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            adv_holder,
            text="☰ 高级选项（学习率/数据增强/设备等，留空使用默认值）",
            variable=var_adv_show,
            command=lambda: self._toggle_adv(key, var_adv_show.get())
        ).pack(anchor=tk.W)
        adv_frame = ttk.LabelFrame(adv_holder,
                                   text="高级参数（留空 = 使用 ultralytics 默认值）",
                                   padding=6)
        adv_grid = ttk.Frame(adv_frame)
        adv_grid.pack(fill=tk.X)

        def add_adv(key, label, tip):
            var = tk.StringVar()
            row = ttk.Frame(adv_grid)
            ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=12).pack(side=tk.LEFT)
            # ❓ 悬停查看详细说明 (v3.1)
            q = ttk.Label(row, text="❓", foreground="#08c",
                          cursor="question_arrow")
            q.pack(side=tk.LEFT, padx=(4, 0))
            Tooltip(q, ADV_DESCRIPTIONS.get(key, ""))
            ttk.Label(row, text=tip, foreground="#666").pack(
                side=tk.LEFT, padx=4)
            row.pack(fill=tk.X, pady=1)
            return var

        adv_vars = {}
        adv_vars["lr0"] = add_adv("lr0", "lr0 初始学习率:", "默认 0.01")
        adv_vars["lrf"] = add_adv("lrf", "lrf 终值学习率:", "默认 0.01")
        adv_vars["weight_decay"] = add_adv("weight_decay", "weight_decay:",
                                           "权重衰减, 默认 0.0005")
        adv_vars["mosaic"] = add_adv("mosaic", "mosaic:", "马赛克增强 0~1, 默认 1.0")
        adv_vars["fliplr"] = add_adv("fliplr", "fliplr:", "水平翻转概率, 默认 0.5")
        adv_vars["degrees"] = add_adv("degrees", "degrees:", "随机旋转角度, 默认 0.0")
        adv_vars["device"] = add_adv("device", "device 设备:",
                                     "GPU编号如 0 / 0,1, 留空自动")
        adv_vars["seed"] = add_adv("seed", "seed 随机种子:", "默认 0")

        adv_cos_row = ttk.Frame(adv_grid)
        adv_cos = tk.BooleanVar(value=True)
        ttk.Checkbutton(adv_cos_row, text="cos_lr 余弦退火调度 (默认开启)",
                        variable=adv_cos).pack(side=tk.LEFT)
        adv_cos_row.pack(fill=tk.X, pady=1)
        self._adv_frames[key] = (adv_frame, var_adv_show)

        yrow = ttk.Frame(lf)
        yrow.pack(fill=tk.X, pady=2)
        ttk.Label(yrow, text="数据集配置:").pack(side=tk.LEFT)
        var_yaml = tk.StringVar(value="")
        ttk.Label(yrow, textvariable=var_yaml, foreground="#06c",
                  wraplength=650).pack(side=tk.LEFT, padx=4)

        bf = ttk.Frame(f, padding=(0, 8))
        bf.pack(fill=tk.X)
        btn_start = ttk.Button(bf, text="开始训练 ▶")
        btn_start.pack(side=tk.LEFT)
        btn_stop = ttk.Button(bf, text="■ 停止训练", state=tk.DISABLED)
        btn_stop.pack(side=tk.LEFT, padx=6)
        btn_open = ttk.Button(bf, text="打开结果文件夹",
                              command=lambda: self.open_result_dir())
        btn_open.pack(side=tk.RIGHT)
        self.start_buttons.append(btn_start)
        btn_stop.configure(command=self.runner.stop)
        self.stop_buttons[key] = btn_stop

        setattr(self, f"{key}_vars", {
            "model_cbs": model_cbs, "custom": var_custom,
            "epochs": v_epochs, "imgsz": v_imgsz,
            "batch": v_batch, "patience": v_patience, "copy_paste": v_cp,
            "workers": v_workers, "cache": v_cache, "yaml": var_yaml,
            "adv_show": var_adv_show, "adv": adv_vars, "adv_cos": adv_cos})
        btn_start.configure(command=lambda: self.on_train(key))
        self.train_tabs_meta[key] = yaml_kind

    def _toggle_adv(self, key, show):
        """高级选项区 展开/收起"""
        adv_frame, _ = self._adv_frames[key]
        if show:
            adv_frame.pack(fill=tk.X, pady=(4, 0))
        else:
            adv_frame.pack_forget()

    def _collect_adv(self, v):
        """收集高级参数 dict；数值留空=不传(用默认)，填了才带上。返回 (adv, 错误信息)"""
        adv = {}
        for k, var in v["adv"].items():
            s = var.get().strip()
            if not s:
                continue
            try:
                if k == "device":
                    adv[k] = s
                elif k == "seed":
                    adv[k] = int(s)
                else:
                    adv[k] = float(s)
            except ValueError:
                return None, f"{k} 不是合法数值: {s}"
        if v["adv_cos"].get():
            adv["cos_lr"] = True
        else:
            adv["cos_lr"] = False
        return adv, None

    def _adv_cmd_args(self, adv):
        """高级参数字典 → 命令行参数列表（值为 None/空的字段不传，用默认）"""
        args = []
        for k in ("lr0", "lrf", "weight_decay", "mosaic", "fliplr", "degrees",
                  "device", "seed"):
            val = adv.get(k)
            if val is None or val == "":
                continue
            args += [f"--{k}", str(val)]
        if "cos_lr" in adv:
            args += ["--cos_lr", "1" if adv["cos_lr"] else "0"]
        return args

    def _collect_models(self, v):
        """收集勾选的预设模型 + 自定义输入，去重保序"""
        models = [m for m, var in v["model_cbs"] if var.get()]
        for x in v["custom"].get().split(","):
            x = x.strip()
            if x and x not in models:
                models.append(x)
        return models

    def _yaml_path_for(self, kind):
        root = self.p_root()
        if not root:
            return ""
        sub = "dataset" if kind == "dataset" else "final_dataset"
        base = "dataset.yaml" if kind == "dataset" else "final_dataset.yaml"
        return os.path.join(root, sub, base)

    def refresh_yaml_labels(self):
        for key in self.train_tabs_meta:
            getattr(self, f"{key}_vars")["yaml"].set(
                self._yaml_path_for(self.train_tabs_meta[key]))

    def _on_root_changed(self):
        """项目根目录变更：刷新派生路径显示与数量预览"""
        self.refresh_yaml_labels()
        self._update_preview2()
        self._update_preview6()

    # —— 数量实时预览 ——

    def _update_preview2(self):
        root = self.p_root()
        if not root or not os.path.isdir(os.path.join(root, "images_all")):
            self.var_preview2.set("（完成抽帧后自动显示预计抽取数量）")
            return
        n = len(core.list_images(os.path.join(root, "images_all")))
        sample = float(self.var_sample.get())
        inner = float(self.var_inner.get())
        s = max(1, int(n * sample))
        t = max(1, int(s * inner))
        self.var_preview2.set(
            f"当前 images_all 共 {n} 张 → 预计抽取 {s} 张"
            f"（train {t} / val {s - t}，剩余 {n - s} 张留待预测）")

    def _update_preview6(self):
        root = self.p_root()
        if not root:
            self.var_preview6.set("")
            return
        imgs = os.path.join(root, "images_all")
        lbls = os.path.join(root, "rest_labels")
        n = len(core.list_images(imgs)) if os.path.isdir(imgs) else 0
        pairs, missing = ([], [])
        if n:
            pairs, missing = core.pair_images_labels(imgs, lbls)
        rt, rv = float(self.var_rtrain.get()), float(self.var_rval.get())
        total = len(pairs)
        tr, va = int(total * rt), int(total * rv)
        self.var_preview6.set(
            f"当前配对 {total} 对 / 缺标签 {len(missing)} 张 → "
            f"预计 train {tr} / val {va} / test {total - tr - va}")

    def on_train(self, key):
        root = self.require_root()
        if root is None:
            return
        if not self.validate_python_exe():
            return
        kind = self.train_tabs_meta[key]
        sub = "dataset" if kind == "dataset" else "final_dataset"
        ds_root = os.path.join(root, sub)
        yaml_path = self._yaml_path_for(kind)
        classes_txt = os.path.join(ds_root, "classes.txt")

        names = core.read_classes(classes_txt) if os.path.isfile(classes_txt) else []
        if not names:
            messagebox.showerror("错误", f"找不到有效的 classes.txt:\n{classes_txt}"
                                         "\n\n请先完成步骤②")
            return
        img_train = os.path.join(ds_root, "images", "train")
        img_val = os.path.join(ds_root, "images", "val")
        if not os.path.isdir(img_train) or not core.list_images(img_train):
            messagebox.showerror("错误", f"训练图片目录为空:\n{img_train}")
            return
        if not os.path.isdir(img_val) or not core.list_images(img_val):
            messagebox.showerror("错误", f"验证图片目录为空:\n{img_val}")
            return

        # 每次开训前自动重建 yaml（单一事实源是 classes.txt）
        try:
            yaml_path = core.build_yaml(ds_root, names, yaml_path)
        except core.JobError as e:
            messagebox.showerror("错误", str(e))
            return

        v = getattr(self, f"{key}_vars")
        models = self._collect_models(v)
        if not models:
            messagebox.showerror("错误", "请至少勾选一个模型，或在自定义栏填写模型名")
            return
        adv, adv_err = self._collect_adv(v)
        if adv_err:
            messagebox.showerror("错误", f"高级参数格式错误:\n{adv_err}\n\n"
                                         "数值类请填数字（如 0.01），留空则使用默认值")
            return
        self.save_config_from_widgets()
        # 多模型对比批次记录
        self._train_batch = [{"model": m, "save_dir": "", "elapsed": 0.0,
                              "ok": False} for m in models]
        self._batch_running = True
        self.status_var.set(f"训练中 {models[0]} (1/{len(models)})")
        self.set_busy(True, key)
        self.log(f"▶ 加入训练队列 {len(models)} 个模型: {', '.join(models)}")
        self.log(f"  数据集: {yaml_path}")
        adv_args = self._adv_cmd_args(adv)
        for i, model in enumerate(models, 1):
            cmd = [self.cfg["python_exe"],
                   os.path.join(APP_DIR, "tool_core.py"),
                   "--job", "train",
                   "--data", yaml_path, "--model", model,
                   "--project_root", root,
                   "--epochs", str(v["epochs"].get()),
                   "--imgsz", str(v["imgsz"].get()),
                   "--batch", str(v["batch"].get()),
                   "--patience", str(v["patience"].get()),
                   "--copy_paste", str(v["copy_paste"].get()),
                   "--workers", str(v["workers"].get()),
                   "--cache_ram", "1" if v["cache"].get() else "0"]
            cmd += adv_args
            self.runner.start(cmd, desc=f"训练 {model} ({i}/{len(models)})",
                              on_done=self._on_train_job_done)

    def _on_train_job_done(self):
        """单个模型训练收尾：记录完成状态（save_dir 已在 SAVE_DIR 行记录）"""
        pass  # 完成状态在 _on_job_exit 统一处理

    def _print_training_summary(self):
        """多模型训练结束后输出对比汇总表"""
        batch = self._train_batch or []
        self.log("=" * 64)
        self.log("多模型训练对比汇总:")
        self.log(f"{'模型':<16}{'best mAP50':>12}{'mAP50-95':>12}{'用时':>10}")
        for item in batch:
            m = item["model"]
            if not item["ok"] or not item["save_dir"]:
                self.log(f"{m:<16}{'失败/未完成':>34}")
                continue
            r = core.read_results_summary(item["save_dir"])
            if not r:
                self.log(f"{m:<16}  (无 results.csv)")
                continue
            mAP50 = r.get("metrics/mAP50(B)")
            mAP95 = r.get("metrics/mAP50-95(B)")
            try:
                self.log(f"{m:<16}{mAP50:>12.3f}{mAP95:>12.3f}"
                         f"{item['elapsed']:>9.1f}s")
            except (TypeError, ValueError):
                self.log(f"{m:<16}  (指标读取异常)")
        self.log("=" * 64)
        self.log("提示: 结果目录已保存各模型 best.pt 与曲线，可点『打开结果文件夹』查看")

    def open_result_dir(self):
        root = self.require_root()
        if root is None:
            return
        d = os.path.join(root, "result")
        if not os.path.isdir(d):
            messagebox.showinfo("提示", f"还没有训练结果目录:\n{d}")
            return
        os.startfile(d)

    # ---------- 页签⑤ 预测剩余80% ---------- #

    def _build_tab5(self, parent):
        f = ttk.Frame(parent, padding=10)
        parent.add(f, text="⑤ 预测剩余80%")
        lf = ttk.LabelFrame(f, text="预测设置", padding=8)
        lf.pack(fill=tk.X)

        row1 = ttk.Frame(lf)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="权重 best.pt:", width=14).pack(side=tk.LEFT)
        self.var_weights = tk.StringVar()
        self.weights_cb = ttk.Combobox(row1, textvariable=self.var_weights,
                                       width=52)
        self.weights_cb.pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="刷新", command=self.refresh_best_pt).pack(
            side=tk.LEFT)
        ttk.Button(row1, text="浏览…",
                   command=lambda: self._browse_file(self.var_weights)).pack(
            side=tk.LEFT, padx=4)

        row2 = ttk.Frame(lf)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="conf 阈值:", width=14).pack(side=tk.LEFT)
        self.var_conf = tk.DoubleVar(value=0.25)
        ttk.Spinbox(row2, from_=0.01, to=0.95, increment=0.01, width=8,
                    textvariable=self.var_conf).pack(side=tk.LEFT)
        ttk.Label(row2, text="(置信度低于此值的框丢弃)").pack(
            side=tk.LEFT, padx=6)

        bf = ttk.Frame(f, padding=(0, 8))
        bf.pack(fill=tk.X)
        self.btn_predict = ttk.Button(bf, text="开始预测 ▶",
                                      command=self.on_predict)
        self.btn_predict.pack(side=tk.LEFT)
        self.start_buttons.append(self.btn_predict)
        self.btn_stop_pred = ttk.Button(bf, text="■ 停止", state=tk.DISABLED,
                                        command=self.runner.stop)
        self.btn_stop_pred.pack(side=tk.LEFT, padx=6)
        self.stop_buttons["predict"] = self.btn_stop_pred
        ttk.Button(bf, text="打开输出文件夹",
                   command=self.open_predict_dir).pack(side=tk.RIGHT)

        tip = ("说明：对 <项目根>/images_all 里剩余的图片预测，框坐标 txt 自动复制到 "
               "<项目根>/rest_labels/。\n之后进入步骤⑥用 labelImg 微调这些标签。"
               "注意：没检出目标的图片不会有 txt，最终划分时将被跳过。")
        ttk.Label(f, text=tip, foreground="#666", wraplength=800,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=6)

    def refresh_best_pt(self):
        root = self.p_root()
        if not root or not os.path.isdir(root):
            messagebox.showerror("提示", "请先填写正确的项目根目录")
            return
        pts = core.collect_best_pt(root)
        self.weights_cb["values"] = pts
        if pts:
            self.var_weights.set(pts[0])
            self.log(f"发现 {len(pts)} 个 best.pt，最新: {pts[0]}")
        else:
            self.var_weights.set("")
            self.log("[注意] 项目根下没有发现 best.pt，请用浏览…手动选权重文件")

    def open_predict_dir(self):
        root = self.require_root()
        if root is None:
            return
        d = os.path.join(root, "predict_out", "predict1")
        if not os.path.isdir(d):
            messagebox.showinfo("提示", f"还没有预测输出:\n{d}")
            return
        os.startfile(d)

    def _browse_file(self, var):
        p = filedialog.askopenfilename(parent=self, title="选择权重文件",
                                       filetypes=[("PyTorch权重", "*.pt"),
                                                  ("所有文件", "*.*")])
        if p:
            var.set(os.path.normpath(p))

    def on_predict(self):
        root = self.require_root()
        if root is None:
            return
        if not self.validate_python_exe():
            return
        weights = self.var_weights.get().strip()
        if not weights or not os.path.isfile(weights):
            messagebox.showerror("错误", f"权重文件不存在:\n{weights}\n\n"
                                         "点【刷新】自动搜索或【浏览…】手动选")
            return
        source = os.path.join(root, "images_all")
        if not core.list_images(source):
            messagebox.showerror("错误", f"没有待预测图片:\n{source}")
            return
        predict_project = os.path.join(root, "predict_out")
        conf = float(self.var_conf.get())
        self.save_config_from_widgets()
        self.status_var.set("正在预测…")
        self.set_busy(True, "predict")
        self.log(f"▶ 开始预测: {source} (conf={conf})")
        self.runner.start(
            [self.cfg["python_exe"], os.path.join(APP_DIR, "tool_core.py"),
             "--job", "predict", "--weights", weights, "--source", source,
             "--conf", str(conf),
             "--project", predict_project, "--name", "predict1"],
            on_done=lambda: self._after_predict(root))

    def _after_predict(self, root):
        src_labels = os.path.join(root, "predict_out", "predict1", "labels")
        dst = os.path.join(root, "rest_labels")
        n = core.post_predict_copy(src_labels, dst)
        self.log(f"✔ 已复制 {n} 个预测txt → {dst}")

    # ---------- 页签⑥ 微调+最终划分 ---------- #

    def _build_tab6(self, parent):
        f = ttk.Frame(parent, padding=10)
        parent.add(f, text="⑥ 微调+最终划分")
        lf = ttk.LabelFrame(f, text="第1步: 用 labelImg 微调预测标签", padding=8)
        lf.pack(fill=tk.X)
        self.btn_lbl_rest = ttk.Button(lf, text="打开 rest 标注 ▶",
                                       command=self.on_labelimg_rest)
        self.btn_lbl_rest.pack(side=tk.LEFT)
        self.start_buttons.append(self.btn_lbl_rest)
        ttk.Label(lf, text="（加载 images_all 图片 + rest_labels 标签）",
                  foreground="#666").pack(side=tk.LEFT, padx=6)

        lf2 = ttk.LabelFrame(f, text="第2步: 最终划分（复制模式，原件不动）", padding=8)
        lf2.pack(fill=tk.X, pady=8)
        row = ttk.Frame(lf2)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="train 比例:").pack(side=tk.LEFT)
        self.var_rtrain = tk.DoubleVar(value=0.7)
        ttk.Spinbox(row, from_=0.1, to=0.9, increment=0.05, width=6,
                    textvariable=self.var_rtrain).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="val 比例:").pack(side=tk.LEFT, padx=(12, 0))
        self.var_rval = tk.DoubleVar(value=0.2)
        ttk.Spinbox(row, from_=0.05, to=0.5, increment=0.05, width=6,
                    textvariable=self.var_rval).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="test = 余数").pack(side=tk.LEFT, padx=6)

        bg_row = ttk.Frame(lf2)
        bg_row.pack(fill=tk.X, pady=2)
        self.var_empty_bg = tk.BooleanVar(value=False)
        ttk.Checkbutton(bg_row, text="无检出图片当背景图纳入（生成空标签）",
                        variable=self.var_empty_bg).pack(side=tk.LEFT)
        ttk.Label(bg_row, text="⚠ 风险: 若图中实际有目标但漏检，会被教成背景",
                  foreground="#c0392b").pack(side=tk.LEFT, padx=6)

        self.var_preview6 = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.var_preview6, foreground="#06c",
                  wraplength=800, justify=tk.LEFT).pack(anchor=tk.W, pady=2)
        self.var_rtrain.trace_add("write", lambda *a: self._update_preview6())
        self.var_rval.trace_add("write", lambda *a: self._update_preview6())

        bf = ttk.Frame(f, padding=(0, 4))
        bf.pack(fill=tk.X)
        self.btn_split = ttk.Button(bf, text="执行最终划分 ▶",
                                    command=self.on_final_split)
        self.btn_split.pack(side=tk.LEFT)
        self.start_buttons.append(self.btn_split)
        ttk.Button(bf, text="打开 final_dataset",
                   command=self.open_final_dir).pack(side=tk.RIGHT)

        tip = ("说明：将 images_all + rest_labels 的配对样本复制划分为 "
               "<项目根>/final_dataset/{images,labels}/{train,val,test}。\n"
               "无标签的图片将被跳过并列出清单；比例之和必须 ≤ 100%。"
               "完成后进入步骤⑦二次训练。")
        ttk.Label(f, text=tip, foreground="#666", wraplength=800,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=6)

    def on_labelimg_rest(self):
        root = self.require_root()
        if root is None:
            return
        img_dir = os.path.join(root, "images_all")
        lbl_dir = os.path.join(root, "rest_labels")
        classes_txt = os.path.join(root, "dataset", "classes.txt")
        if not os.path.isdir(img_dir):
            messagebox.showerror("错误", f"图片目录不存在:\n{img_dir}")
            return
        if not os.path.isfile(classes_txt):
            messagebox.showerror("错误", f"找不到 classes.txt:\n{classes_txt}\n\n"
                                         "请先完成步骤②")
            return
        os.makedirs(lbl_dir, exist_ok=True)
        self._launch_labelimg(img_dir, lbl_dir, classes_txt, "rest微调")

    def on_final_split(self):
        root = self.require_root()
        if root is None:
            return
        imgs = os.path.join(root, "images_all")
        lbls = os.path.join(root, "rest_labels")
        out = os.path.join(root, "final_dataset")
        rt, rv = float(self.var_rtrain.get()), float(self.var_rval.get())
        if rt <= 0 or rv <= 0 or rt + rv > 1.0 + 1e-9:
            messagebox.showerror(
                "错误", f"比例非法: train={rt:g} val={rv:g} test={1 - rt - rv:g}，"
                        "两项都必须大于0且总和不超过100%")
            return
        pairs, missing = ([], [])
        if os.path.isdir(imgs):
            pairs, missing = core.pair_images_labels(imgs, lbls)
        if not pairs:
            messagebox.showerror("错误", f"没有『图片+标签』配对:\n{imgs}\n\n"
                                         "请先完成步骤⑤预测得到 rest_labels")
            return
        total = len(pairs)
        tr = int(total * rt)
        va = int(total * rv)
        te = total - tr - va
        if not messagebox.askyesno(
                "确认划分",
                f"共 {total} 个有效配对，将【复制】划分为:\n"
                f"  train {tr} / val {va} / test {te}\n"
                f"  另有 {len(missing)} 张无标签图片跳过\n"
                f"原件不动，输出到 final_dataset/\n\n确定执行？"):
            return
        self.save_config_from_widgets()
        self.status_var.set("正在最终划分…")
        self.set_busy(True)
        empty_bg = bool(self.var_empty_bg.get())
        self.log(f"▶ 最终划分: {imgs} + {lbls} → {out} ({rt:g}/{rv:g}/余)"
                 + ("，无检出图当背景纳入" if empty_bg else ""))

        def work():
            code = 0
            core.log_hook = lambda m: self.root.after(0, self.log, m)
            try:
                core.run_split(imgs, lbls, out, rt, rv, empty_bg=empty_bg)
            except core.JobError as e:
                self.log(f"JOB_ERROR: {e}", error=True)
                code = 2
            except Exception:
                import traceback
                self.log(traceback.format_exc(), error=True)
                code = 1
            finally:
                core.log_hook = None
            self.root.after(0, lambda: self._finish_light(code, "最终划分"))

        threading.Thread(target=work, daemon=True).start()

    def open_final_dir(self):
        root = self.require_root()
        if root is None:
            return
        d = os.path.join(root, "final_dataset")
        if not os.path.isdir(d):
            messagebox.showinfo("提示", f"还没有最终数据集:\n{d}")
            return
        os.startfile(d)

    # ---------- JobRunner 回调 ---------- #

    def _on_job_line(self, line):
        if line.startswith("PROGRESS"):
            self.status_var.set(line)
            return
        if line.startswith("SAVE_DIR:"):
            save_dir = line.split(":", 1)[1].strip()
            self._last_save_dir = save_dir
            # 记录到多模型对比批次中的当前模型
            if self._train_batch and self.runner.desc.startswith("训练 "):
                model = self.runner.desc.split(" (")[0][len("训练 "):].strip()
                for item in self._train_batch:
                    if item["model"] == model and not item["save_dir"]:
                        item["save_dir"] = save_dir
                        break
            return
        if "JOB_ERROR:" in line:
            self.log(line, error=True)
            return
        self.log(line)

    def _on_job_exit(self, code):
        # —— 训练队列批次：记录完成情况，队列没空则继续（保持 busy）——
        if self._train_batch and self.runner.desc.startswith("训练 "):
            model = self.runner.desc.split(" (")[0][len("训练 "):].strip()
            for item in self._train_batch:
                if item["model"] == model and not item["ok"]:
                    item["ok"] = (code == 0)
                    item["elapsed"] = self.runner.last_elapsed
                    break
            if code == 0:
                self.log(f"✔ {model} 训练完成 ({self.runner.last_elapsed:.1f}s)")
            else:
                self.log(f"✘ {model} 训练{'被停止' if self.runner.stopped_by_user else '失败'}"
                         f" (exit={code})", error=True)
            if self.runner.pending:
                nxt = self.runner.pending[0][1]
                pos = self.runner.queue_len
                total = len(self._train_batch)
                self.status_var.set(f"训练中 {nxt} ({pos}/{total})")
                self.log(f"队列中下一个: {nxt} ({pos}/{total})")
                return   # 队列继续，按钮保持禁用

        # —— 常规任务 / 批次结束 ——
        self.set_busy(False)
        self.status_var.set("空闲")
        if code == 0:
            self.log("✔ 任务完成")
            hook = getattr(self.runner, "_on_done", None)
            if hook:
                try:
                    hook()
                except Exception as e:
                    self.log(f"[警告] 收尾操作出错: {e}", error=True)
            if self._last_save_dir:
                self.log(f"产物目录: {self._last_save_dir}")
        else:
            reason = "被用户停止" if self.runner.stopped_by_user else \
                     f"失败 (exit={code})"
            self.log(f"✘ 任务{reason}", error=True)

        # 多模型批次全部结束 → 输出对比汇总
        if self._train_batch:
            self._print_training_summary()
            self._train_batch = None
            self._batch_running = False

    # ---------- 关闭保护 ---------- #

    def on_close(self):
        if self.runner.is_running:
            if not messagebox.askokcancel(
                    "任务运行中", "后台任务还在运行，确定退出？\n"
                                 "(退出会终止当前任务进程)"):
                return
            self.runner.stop()
        self.save_config_from_widgets()
        self.destroy()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
