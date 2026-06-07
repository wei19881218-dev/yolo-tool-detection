# demo/lab_tools_video_demo_25s.py
# 使用微调后的 YOLO-World 模型进行视频/摄像头检测，并保证输出视频至少指定秒数
#
# 视频文件，至少输出 25 秒：
#   python demo/lab_tools_video_demo_25s.py configs/custom/lab_tools_finetune_7_2_1.py work_dirs/lab_tools_finetune_7_2_1/best_coco_bbox_mAP_epoch_45.pth --source input1.mp4 --out outputs/lab_tools_result_25s.mp4 --min-seconds 25 --fps-out 25
#
# 如果原视频不足 25 秒，脚本会从头循环视频帧直到输出够 25 秒。
# 如果原视频超过 25 秒，默认会处理完整视频；如果你只想处理前 25 秒，增加 --max-seconds 25。
#
# 只处理前 25 秒：
#   python demo/lab_tools_video_demo_25s.py configs/custom/lab_tools_finetune_7_2_1.py work_dirs/lab_tools_finetune_7_2_1/best_coco_bbox_mAP_epoch_45.pth --source input1.mp4 --out outputs/lab_tools_result_25s.mp4 --min-seconds 25 --max-seconds 25 --fps-out 25

import argparse
import time
from pathlib import Path

import cv2
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector

CLASS_NAMES = [
    "screwdriver",
    "breadboard",
    "hex_key",
    "plier",
    "jumper_wire_bundle",
    "wrench",
    "soldering_iron",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YOLO-World config path")
    parser.add_argument("checkpoint", help="fine-tuned checkpoint path")
    parser.add_argument("--source", default="0", help="video path or camera index, e.g. 0 / input.mp4")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--score-thr", type=float, default=0.25, help="score threshold")
    parser.add_argument("--out", default="", help="output video path, e.g. outputs/result.mp4")
    parser.add_argument("--show", action="store_true", help="show realtime window")
    parser.add_argument("--fps-out", type=float, default=25.0, help="output video FPS")
    parser.add_argument("--min-seconds", type=float, default=0.0, help="minimum output duration in seconds; 0 means disabled")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="maximum output duration in seconds; 0 means process full video")
    return parser.parse_args()


def build_pipeline(cfg):
    pipeline = cfg.test_pipeline.copy()
    if pipeline[0]["type"] in ["LoadImageFromFile", "mmdet.LoadImageFromFile"]:
        pipeline[0] = dict(type="mmdet.LoadImageFromNDArray")
    return Compose(pipeline)


def draw_result(frame, result, score_thr=0.25):
    pred = result.pred_instances
    if pred is None or len(pred) == 0:
        return frame

    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()

    for bbox, score, label in zip(bboxes, scores, labels):
        if score < score_thr:
            continue

        x1, y1, x2, y2 = bbox.astype(int).tolist()
        x1 = max(0, min(x1, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        x2 = max(0, min(x2, frame.shape[1] - 1))
        y2 = max(0, min(y2, frame.shape[0] - 1))

        name = CLASS_NAMES[int(label)] if 0 <= int(label) < len(CLASS_NAMES) else str(int(label))
        text = f"{name} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, text, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return frame


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()
    model.dataset_meta = {"classes": CLASS_NAMES}

    test_pipeline = build_pipeline(cfg)
    texts = [[name] for name in CLASS_NAMES]

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源：{args.source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 0 or src_fps > 240:
        src_fps = 25.0

    out_fps = args.fps_out if args.fps_out and args.fps_out > 0 else src_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_duration = total_frames / src_fps if total_frames > 0 else 0

    print(f"输入视频 FPS: {src_fps:.3f}")
    print(f"输入视频总帧数: {total_frames}")
    print(f"输入视频时长: {src_duration:.2f} 秒")
    print(f"输出视频 FPS: {out_fps:.3f}")

    min_frames = int(args.min_seconds * out_fps) if args.min_seconds > 0 else 0
    max_frames = int(args.max_seconds * out_fps) if args.max_seconds > 0 else 0

    writer = None
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height))

    frame_id = 0
    output_frames = 0
    loop_count = 0
    t0 = time.time()

    with torch.no_grad():
        while True:
            if max_frames > 0 and output_frames >= max_frames:
                break

            ret, frame = cap.read()

            if not ret:
                if min_frames > 0 and output_frames < min_frames:
                    loop_count += 1
                    print(f"原视频已结束，但输出不足 {args.min_seconds} 秒，开始第 {loop_count} 次循环补帧。")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_id = 0
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    break

            data_info = dict(img=frame, img_id=frame_id, texts=texts)
            data = test_pipeline(data_info)
            data["inputs"] = [data["inputs"]]
            data["data_samples"] = [data["data_samples"]]

            result = model.test_step(data)[0]
            vis_frame = draw_result(frame.copy(), result, args.score_thr)

            output_frames += 1
            frame_id += 1

            elapsed = time.time() - t0
            cur_fps = output_frames / elapsed if elapsed > 0 else 0
            cv2.putText(vis_frame, f"FPS: {cur_fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            if writer is not None:
                writer.write(vis_frame)

            if args.show:
                cv2.imshow("YOLO-World Lab Tools Detection", vis_frame)
                if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                    break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    out_duration = output_frames / out_fps if out_fps > 0 else 0
    print(f"完成，共输出 {output_frames} 帧，输出时长约 {out_duration:.2f} 秒。")
    if args.out:
        print(f"结果视频已保存到：{args.out}")


if __name__ == "__main__":
    main()
