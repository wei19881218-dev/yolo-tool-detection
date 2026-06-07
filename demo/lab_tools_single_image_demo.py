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

# 关键：注册 YOLO-World 自定义模块
import yolo_world  # noqa: F401


LAB_TOOL_CLASSES = [
    "screwdriver",
    "breadboard",
    "hex key",
    "plier",
    "jumper wire bundle",
    "wrench",
    "soldering iron",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO-World original model single image demo for lab tools"
    )
    parser.add_argument("config", help="YOLO-World config path")
    parser.add_argument("checkpoint", help="Original YOLO-World checkpoint path")
    parser.add_argument("image", help="Input image path")
    parser.add_argument(
        "--classes",
        default=",".join(LAB_TOOL_CLASSES),
        help="Class prompts separated by comma, or a txt file with one class per line",
    )
    parser.add_argument("--out", default="outputs/original_yoloworld_lab_tools.jpg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.20)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--box-thickness", type=int, default=5)
    parser.add_argument("--text-scale", type=float, default=0.7)
    parser.add_argument("--text-thickness", type=int, default=2)
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, e.g. model.backbone.text_model.model_name=xxx",
    )
    return parser.parse_args()


def load_classes(classes_arg):
    if classes_arg.endswith(".txt"):
        with open(classes_arg, "r", encoding="utf-8") as f:
            names = [x.strip() for x in f.readlines() if x.strip()]
    else:
        names = [x.strip() for x in classes_arg.split(",") if x.strip()]
    return names


def build_test_pipeline():
    return Compose([
        dict(type="LoadImageFromFile"),
        dict(type="YOLOv5KeepRatioResize", scale=(640, 640)),
        dict(
            type="LetterResize",
            scale=(640, 640),
            allow_scale_up=False,
            pad_val=dict(img=114),
        ),
        dict(type="LoadText"),
        dict(
            type="mmdet.PackDetInputs",
            meta_keys=(
                "img_id",
                "img_path",
                "ori_shape",
                "img_shape",
                "scale_factor",
                "pad_param",
                "texts",
            ),
        ),
    ])


def build_model(config, checkpoint, device, cfg_options=None, num_classes=None):
    cfg = Config.fromfile(config)

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    cfg.load_from = checkpoint
    cfg.model.train_cfg = None

    if num_classes is not None:
        cfg.model.num_test_classes = num_classes

    if "backbone" in cfg.model and "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    init_default_scope(cfg.get("default_scope", "mmdet"))

    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location="cpu")

    model.cfg = cfg
    model.to(device)
    model.eval()
    return model


def draw_results(image_path, bboxes, labels, scores, class_names, out_path,
                 box_thickness=5, text_scale=0.7, text_thickness=2):
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
            text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
        )
        y_text = max(y1, th + baseline + 3)

        cv2.rectangle(
            img,
            (x1, y_text - th - baseline - 4),
            (x1 + tw + 4, y_text + baseline),
            color,
            -1,
        )
        cv2.putText(
            img,
            text,
            (x1 + 2, y_text - 3),
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

    class_names = load_classes(args.classes)
    texts = [[name] for name in class_names] + [[" "]]

    model = build_model(
        args.config,
        args.checkpoint,
        args.device,
        cfg_options=args.cfg_options,
        num_classes=len(class_names),
    )

    test_pipeline = build_test_pipeline()

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

    with autocast(enabled=args.amp), torch.no_grad():
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
            f"{i:02d}: {class_names[int(label)]} "
            f"score={float(score):.3f} "
            f"box={box.astype(int).tolist()}"
        )

    draw_results(
        args.image,
        bboxes,
        labels,
        scores,
        class_names,
        args.out,
        box_thickness=args.box_thickness,
        text_scale=args.text_scale,
        text_thickness=args.text_thickness,
    )

    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()