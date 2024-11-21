import argparse
from pathlib import Path
from yolo_sod.inference import process as inference_process
from yolo_sod.export import process as export_process
from yolo_sod.image_processor import process as image_processor
import torch
import os


def inference():
    """Inference process."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output JSON file"
    )
    parser.add_argument("--model-od", type=str, required=True, help="Path to OD model")
    parser.add_argument(
        "--object-categories",
        type=str,
        nargs="+",
        required=False,
        choices=["person", "car", "bicycle", "bus", "motorcycle", "truck"],
        default=["person", "car", "bicycle", "bus", "motorcycle", "truck"],
        help="List of object categories to detect",
    )
    parser.add_argument(
        "--confidence-threshold-od",
        type=float,
        default=0.05,
        help="Confidence threshold for OD model",
    )
    parser.add_argument(
        "--nms-iou-threshold", type=float, default=0.3, help="NMS IOU threshold"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose export information",
        default=False,
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=2,
        help="Size of the queue for the filler process",
    )

    args = parser.parse_args()
    if Path(args.input).is_dir():
        image_processor(args)
    else:
        inference_process(args)


def export():
    """Export process."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-od", type=str, default="yolo11l.pt", help="OD model base to export"
    )
    parser.add_argument(
        "--precision-od",
        type=str,
        default="INT8",
        help="Precision of the exported model",
    )
    parser.add_argument("--opset", type=int, help="ONNX opset version")
    # NOTE workspace should be a float value, but a bug in Ultralytics' code causes a type error
    parser.add_argument(
        "--workspace", type=int, default=2, help="TensorRT workspace size (GiB)"
    )
    parser.add_argument(
        "--device", type=str, default="0", help="GPU device to use for export"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="engine",
        help="Export format (engine, onnx)",
        choices=["engine", "onnx"],
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset to use for INT8 calibration",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[480, 864],
        help="Image size for the exported model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for the exported model"
    )

    args = parser.parse_args()
    export_process(args)

    # get name of the gpu device and replace spaces with underscores
    device_index = int(args.device)
    device_name = torch.cuda.get_device_name(device_index).replace(" ", "_")

    # set the output base name: model_name + device_name + precision + input_size
    model_base = args.model_od.split(".")[0]
    output_name = (
        model_base
        + "-"
        + str(args.img_size[0])
        + "x"
        + str(args.img_size[1])
    )
    
    engine_name = f"{output_name  + '-' + device_name + '-' + args.precision_od}"

    # rename files {model_base} to {output_name}
    try:
        os.rename(f"{model_base}.onnx", f"{output_name}.onnx")
        os.rename(f"{model_base}.engine", f"{engine_name}.engine")
        os.rename(f"{model_base}.cache", f"{output_name}.cache")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    inference()
