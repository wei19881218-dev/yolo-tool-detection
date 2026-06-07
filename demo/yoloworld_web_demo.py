# -*- coding: utf-8 -*-
"""
YOLO-World Web Demo: image/video open-vocabulary detection.
Put this file under CODE/demo/web_demo.py, then run from the CODE directory.
"""

import argparse
from datetime import datetime
import os
import os.path as osp
import tempfile
from typing import List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import nms

from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS


BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=4)
MASK_ANNOTATOR = sv.MaskAnnotator()

# 实验室工具类别：界面显示中文+英文，实际送入模型的是英文类别名。
TOOL_LABELS = [
    ("螺丝刀 screwdriver", "screwdriver"),
    ("面包板 breadboard", "breadboard"),
    ("内六角扳手 hex_key", "hex_key"),
    ("钳子 plier", "plier"),
    ("杜邦线 jumper_wire_bundle", "jumper_wire_bundle"),
    ("扳手 wrench", "wrench"),
    ("电烙铁 soldering_iron", "soldering_iron"),
]
TOOL_DISPLAY_CHOICES = [item[0] for item in TOOL_LABELS]
DISPLAY_TO_EN = dict(TOOL_LABELS)
EN_TO_DISPLAY = {en: display for display, en in TOOL_LABELS}


class LabelAnnotator(sv.LabelAnnotator):
    @staticmethod
    def resolve_text_background_xyxy(center_coordinates, text_wh, position):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=8, text_scale=0.9, text_thickness=2)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO-World image/video web demo")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("checkpoint", help="Checkpoint file path")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--work-dir", default="web_outputs", help="Directory for outputs")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", default=8080, type=int)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, e.g. key=value",
    )
    return parser.parse_args()


def split_categories(selection) -> List[str]:
    """
    支持两种输入：
    1. CheckboxGroup 返回的列表，例如 ["螺丝刀 screwdriver"]
    2. 兼容旧版文本输入，例如 "screwdriver,breadboard"
    最终统一转换为英文类别名，供 YOLO-World 文本编码器使用。
    """
    if isinstance(selection, (list, tuple)):
        cats = [DISPLAY_TO_EN.get(str(item).strip(), str(item).strip()) for item in selection if str(item).strip()]
    else:
        raw_items = [t.strip() for t in str(selection).replace("，", ",").split(",") if t.strip()]
        cats = [DISPLAY_TO_EN.get(item, item) for item in raw_items]

    if not cats:
        raise gr.Error("请至少勾选一个类别，例如：螺丝刀、面包板、钳子")
    return cats


def build_runner(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint

    # The demo only needs inference.
    if hasattr(cfg.model, "train_cfg"):
        cfg.model.train_cfg = None

    if "runner_type" not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.to(args.device)
    runner.model.eval()

    pipeline = cfg.test_dataloader.dataset.pipeline
    pipeline[0].type = "mmdet.LoadImageFromNDArray"
    runner.pipeline = Compose(pipeline)
    return runner


def prepare_texts(runner, category_text: str):
    categories = split_categories(category_text)
    texts = [[c] for c in categories] + [[" "]]
    runner.model.dataset_meta = {"classes": categories}
    # Re-parameterize when the user changes vocabulary/prompts.
    runner.model.reparameterize(texts)
    return texts


def predict_instances(runner, image_array, texts, score_thr: float, nms_thr: float, max_boxes: int):
    data_info = dict(img_id=0, img=image_array, texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    if len(pred_instances) > 0:
        keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
        pred_instances = pred_instances[keep]
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances) > max_boxes:
        indices = pred_instances.scores.float().topk(max_boxes)[1]
        pred_instances = pred_instances[indices]

    return pred_instances.cpu().numpy()


def draw_result(image_array, pred_instances, texts):
    masks = pred_instances["masks"] if "masks" in pred_instances else None
    detections = sv.Detections(
        xyxy=pred_instances["bboxes"],
        class_id=pred_instances["labels"],
        confidence=pred_instances["scores"],
        mask=masks,
    )
    labels = []
    for class_id, confidence in zip(detections.class_id, detections.confidence):
        en_name = texts[int(class_id)][0]
        # display_name = EN_TO_DISPLAY.get(en_name, en_name)
        labels.append(f"{en_name} {float(confidence):.2f}")

    annotated = image_array.copy()
    annotated = BOUNDING_BOX_ANNOTATOR.annotate(annotated, detections)
    annotated = LABEL_ANNOTATOR.annotate(annotated, detections, labels=labels)
    if masks is not None:
        annotated = MASK_ANNOTATOR.annotate(annotated, detections)
    return annotated


def load_image_from_file(file_obj):
    """从 gr.File 上传结果中读取图片，用于不点叉号直接替换输入图片。"""
    if file_obj is None:
        return None
    path = file_obj
    if isinstance(file_obj, (list, tuple)) and file_obj:
        path = file_obj[0]
    if hasattr(path, "name"):
        path = path.name
    return Image.open(path).convert("RGB")


def run_image(
    runner,
    image: Optional[Image.Image],
    category_text: str,
    score_thr: float,
    nms_thr: float,
    max_boxes: int,
    history=None,
):
    if image is None:
        raise gr.Error("请先上传一张图片")

    texts = prepare_texts(runner, category_text)
    image_rgb = np.array(image.convert("RGB"))
    pred_instances = predict_instances(runner, image_rgb, texts, score_thr, nms_thr, max_boxes)
    annotated_rgb = draw_result(image_rgb, pred_instances, texts)

    os.makedirs("web_outputs", exist_ok=True)
    out_img = Image.fromarray(annotated_rgb)
    out_path = osp.join("web_outputs", next(tempfile._get_candidate_names()) + ".jpg")
    out_img.save(out_path, quality=95)

    detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history = list(history or [])
    history.insert(0, (out_path, f"{detect_time}｜{len(pred_instances)}个检测框"))
    history = history[:3]

    status = f"✅ 图片检测完成，共输出 {len(pred_instances)} 个检测框。检测时间：{detect_time}"
    return out_img, out_path, status, history, history


def run_video(
    runner,
    video_path: Optional[str],
    category_text: str,
    score_thr: float,
    nms_thr: float,
    max_boxes: int,
    frame_interval: int,
    progress=gr.Progress(),
):
    if video_path is None:
        raise gr.Error("请先上传一个视频")

    texts = prepare_texts(runner, category_text)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("视频读取失败，请检查视频格式")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    os.makedirs("web_outputs", exist_ok=True)
    out_path = osp.join("web_outputs", next(tempfile._get_candidate_names()) + ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    last_pred = None
    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Infer every N frames to make long videos faster; reuse last boxes between sampled frames.
        if idx % max(1, int(frame_interval)) == 0 or last_pred is None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            last_pred = predict_instances(runner, frame_rgb, texts, score_thr, nms_thr, max_boxes)

        annotated_rgb = draw_result(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), last_pred, texts)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        writer.write(annotated_bgr)

        idx += 1
        if frame_count > 0 and idx % 5 == 0:
            progress(idx / frame_count, desc=f"视频处理中：{idx}/{frame_count} 帧")

    cap.release()
    writer.release()
    return out_path



def clear_history():
    return [], []


def launch_demo(runner, args):
    default_classes = TOOL_DISPLAY_CHOICES
    css = """
    #title {text-align: center;}

    .gradio-container {
        max-width: 1180px !important;
        margin: auto !important;
    }

    .main-title {
        text-align: center;
        font-size: 30px;
        font-weight: 900;
        margin-top: 4px;
        margin-bottom: 6px;
        color: #17233c;
        letter-spacing: 0.2px;
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-bottom: 24px;
    }

    .tool-card {
        border: 1px solid #e4e8ef !important;
        border-radius: 12px !important;
        padding: 14px 16px 10px 16px !important;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04) !important;
        margin-bottom: 18px !important;
    }

    .tool-checkbox-group label {
        font-size: 15px !important;
        font-weight: 700 !important;
    }

    .tool-checkbox-group .wrap {
        gap: 10px !important;
    }

    .tool-checkbox-group .wrap > label,
    .tool-checkbox-group label[data-testid="checkbox"] {
        border: 1px solid #e5eaf2 !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        background: #fff !important;
        box-shadow: 0 1px 4px rgba(16, 24, 40, 0.05) !important;
    }

    .panel-title {
        font-size: 18px;
        font-weight: 900;
        margin-bottom: 10px;
        color: #111827;
    }

    .image-card {
        height: 360px !important;
        min-height: 360px !important;
        max-height: 360px !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        margin: 0 !important;
    }

    .image-card > div,
    .image-card .wrap,
    .image-card .container,
    .image-card [data-testid="image"],
    .image-card [data-testid="image-container"] {
        height: 360px !important;
        min-height: 360px !important;
        max-height: 360px !important;
        margin: 0 !important;
    }

    .image-card img,
    .image-card canvas {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain !important;
    }

    .input-image-card {
        border: 2px dashed #ff8a1f !important;
        background: linear-gradient(135deg, #fff6eb, #fffaf4) !important;
    }

    .output-image-card {
        border: 2px solid #2ecc71 !important;
        background: linear-gradient(135deg, #effdf5, #f8fffb) !important;
    }

    .image-card .icon-buttons,
    .image-card .image-buttons,
    .image-card .tools,
    .image-card footer {
        display: none !important;
    }

    /* 尽量隐藏 Gradio 上传框里的提示文字，只保留图标 */
    .input-image-card .upload-text,
    .input-image-card .or,
    .input-image-card .browse,
    .input-image-card p,
    .input-image-card span:not(:has(svg)) {
        font-size: 0 !important;
    }

    .video-card {
        height: 360px !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    .control-row {
        align-items: end !important;
        margin-top: 12px !important;
        margin-bottom: 18px !important;
    }

    .compact-number input {
        text-align: center !important;
        font-weight: 700 !important;
    }

    .main-detect-btn button {
        height: 44px !important;
        font-size: 17px !important;
        font-weight: 900 !important;
        border-radius: 10px !important;
        background: linear-gradient(90deg, #ffbd7a, #ffa45b) !important;
        color: #e65000 !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(255, 149, 59, 0.25) !important;
    }

    .history-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 4px;
        margin-bottom: 10px;
    }

    .history-title {
        font-size: 17px;
        font-weight: 900;
        color: #111827;
    }

    .clear-btn button {
        height: 36px !important;
        border-radius: 9px !important;
        border: 1px solid #ffd2d2 !important;
        background: #fff5f5 !important;
        color: #f04444 !important;
        font-weight: 800 !important;
    }

    .history-gallery {
        border: none !important;
        padding: 0 !important;
        max-height: 330px !important;
        overflow-y: auto !important;
    }

    .history-gallery img {
        object-fit: contain !important;
        border-radius: 8px !important;
    }

    .history-gallery .caption,
    .history-gallery figcaption {
        font-size: 13px !important;
        font-weight: 800 !important;
        color: #1f2937 !important;
    }

    .status-box textarea {
        font-size: 15px !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
    }

    .hint {
        font-size: 14px;
        color: #666;
    }
    """

    with gr.Blocks(title="YOLO-World Web Demo") as demo:
        gr.HTML(
            """
            <div class="main-title">YOLO-World 实验室工具目标检测网页演示</div>
            <div class="subtitle">上传图片或视频，勾选要检测的实验室工具类别。</div>
            """
        )

        with gr.Group(elem_classes=["tool-card"]):
            category_text = gr.CheckboxGroup(
                choices=TOOL_DISPLAY_CHOICES,
                value=default_classes,
                label="检测类别 / Prompts（可多选）",
                elem_classes=["tool-checkbox-group"],
            )

        with gr.Tab("图片检测"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.HTML("<div class='panel-title'>① 输入图片</div>")
                    input_image = gr.Image(
                        type="pil",
                        label=None,
                        show_label=False,
                        height=360,
                        sources=["upload"],
                        elem_classes=["image-card", "input-image-card"],
                    )

                with gr.Column(scale=1):
                    gr.HTML("<div class='panel-title'>② 标注结果</div>")
                    output_image = gr.Image(
                        type="pil",
                        label=None,
                        show_label=False,
                        height=360,
                        interactive=False,
                        elem_classes=["image-card", "output-image-card"],
                    )

            with gr.Row(elem_classes=["control-row"]):
                score_thr = gr.Number(value=0.25, label="置信度阈值", elem_classes=["compact-number"], precision=2)
                nms_thr = gr.Number(value=0.70, label="NMS IoU 阈值", elem_classes=["compact-number"], precision=2)
                max_boxes = gr.Number(value=20, label="最多保留框数", elem_classes=["compact-number"], precision=0)
                image_btn = gr.Button("🚀 开始图片检测", variant="primary", elem_classes=["main-detect-btn"], scale=2)

            image_status = gr.Textbox(
                label="运行状态",
                value="等待上传图片并开始检测。",
                interactive=False,
                visible=False,
                elem_classes=["status-box"],
            )
            download_image_file = gr.File(label="下载标注图片", visible=False, file_count="single")
            history_state = gr.State([])

            with gr.Row(elem_classes=["history-head"]):
                gr.HTML("<div class='history-title'>最近三张标注结果</div>")
                clear_history_btn = gr.Button("🗑 清空历史记录", elem_classes=["clear-btn"], scale=0)

            history_gallery = gr.Gallery(
                label=None,
                show_label=False,
                columns=3,
                height=290,
                elem_classes=["history-gallery"],
            )

            image_btn.click(
                fn=lambda img, text, s, n, m, hist: run_image(runner, img, text, float(s), float(n), int(m), hist),
                inputs=[input_image, category_text, score_thr, nms_thr, max_boxes, history_state],
                outputs=[output_image, download_image_file, image_status, history_state, history_gallery],
            )
            clear_history_btn.click(
                fn=clear_history,
                inputs=None,
                outputs=[history_state, history_gallery],
            )

        with gr.Tab("视频检测"):
            frame_interval = gr.Slider(1, 10, value=1, step=1, label="抽帧间隔：1 表示每帧都检测")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.HTML("<div class='panel-title'>① 输入视频</div>")
                    input_video = gr.Video(label="输入视频", height=360, elem_classes=["video-card"])
                with gr.Column(scale=1):
                    gr.HTML("<div class='panel-title'>② 标注后视频</div>")
                    output_video = gr.Video(label="标注后视频", height=360, elem_classes=["video-card"])

            with gr.Row(elem_classes=["control-row"]):
                video_score_thr = gr.Number(value=0.25, label="置信度阈值", elem_classes=["compact-number"], precision=2)
                video_nms_thr = gr.Number(value=0.70, label="NMS IoU 阈值", elem_classes=["compact-number"], precision=2)
                video_max_boxes = gr.Number(value=20, label="最多保留框数", elem_classes=["compact-number"], precision=0)
                video_btn = gr.Button("🎬 开始视频检测", variant="primary", elem_classes=["main-detect-btn"], scale=2)

            video_btn.click(
                fn=lambda vid, text, s, n, m, fi: run_video(runner, vid, text, float(s), float(n), int(m), fi),
                inputs=[input_video, category_text, video_score_thr, video_nms_thr, video_max_boxes, frame_interval],
                outputs=output_video,
            )

    demo.launch(server_name=args.server_name, server_port=args.server_port, css=css)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    runner = build_runner(args)
    launch_demo(runner, args)
