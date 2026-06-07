import argparse
import os
import os.path as osp

import cv2
import torch
from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.runner.amp import autocast
from mmdet.registry import MODELS

import yolo_world  # noqa: F401


LAB_TOOL_CLASSES = [
    "screwdriver",
    "breadboard",
    "hex_key",
    "plier",
    "jumper_wire_bundle",
    "wrench",
    "soldering_iron",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuned YOLO-World single image detection for lab tools"
    )
    parser.add_argument("config", help="Fine-tuned YOLO-World config path")
    parser.add_argument("checkpoint", help="Fine-tuned checkpoint path")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--out", default="outputs/finetuned_lab_tools_vis.jpg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--box-thickness", type=int, default=6)
    parser.add_argument("--text-scale", type=float, default=0.8)
    parser.add_argument("--text-thickness", type=int, default=2)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options if needed",
    )
    return parser.parse_args()


def build_model(config_path, checkpoint_path, device, cfg_options=None):
    cfg = Config.fromfile(config_path)

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    cfg.load_from = checkpoint_path
    cfg.model.train_cfg = None

    # 微调模型固定 7 类
    cfg.model.num_test_classes = len(LAB_TOOL_CLASSES)

    # 推理时不需要重新初始化 backbone
    if "backbone" in cfg.model and "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    init_default_scope(cfg.get("default_scope", "mmdet"))

    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location="cpu")

    model.cfg = cfg
    model.to(device)
    model.eval()

    return model, cfg


def build_pipeline(cfg):
    return Compose(cfg.test_pipeline)


def draw_results(
    image_path,
    bboxes,
    labels,
    scores,
    class_names,
    out_path,
    box_thickness=6,
    text_scale=0.8,
    text_thickness=2,
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    color = (0, 0, 255)

    for box, label, score in zip(bboxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int).tolist()
        cls_name = class_names[int(label)]
        text = f"{cls_name} {float(score):.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        (tw, th), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            text_thickness,
        )

        y_text = max(y1, th + baseline + 4)

        cv2.rectangle(
            img,
            (x1, y_text - th - baseline - 5),
            (x1 + tw + 6, y_text + baseline),
            color,
            -1,
        )

        cv2.putText(
            img,
            text,
            (x1 + 3, y_text - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


def main():
    args = parse_args()

    model, cfg = build_model(
        args.config,
        args.checkpoint,
        args.device,
        cfg_options=args.cfg_options,
    )

    test_pipeline = build_pipeline(cfg)

    # 这里用带下划线的类别名，保持和微调训练时一致
    texts = [[name] for name in LAB_TOOL_CLASSES] + [[" "]]

    data_info = dict(
        img_id=0,
        img_path=args.image,
        texts=texts,
    )

    data_info = test_pipeline(data_info)

    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with torch.no_grad(), autocast(enabled=args.amp):
        result = model.test_step(data_batch)[0]

    pred = result.pred_instances
    pred = pred[pred.scores.float() > args.score_thr]

    if len(pred.scores) > args.topk:
        keep = pred.scores.float().topk(args.topk)[1]
        pred = pred[keep]

    pred = pred.cpu().numpy()

    bboxes = pred["bboxes"]
    labels = pred["labels"]
    scores = pred["scores"]

    print("\nDetection results:")
    for i, (box, label, score) in enumerate(zip(bboxes, labels, scores), 1):
        print(
            f"{i:02d}: {LAB_TOOL_CLASSES[int(label)]} "
            f"score={float(score):.3f} "
            f"box={box.astype(int).tolist()}"
        )

    draw_results(
        args.image,
        bboxes,
        labels,
        scores,
        LAB_TOOL_CLASSES,
        args.out,
        box_thickness=args.box_thickness,
        text_scale=args.text_scale,
        text_thickness=args.text_thickness,
    )

    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()