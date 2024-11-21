from pathlib import Path
from ultralytics import YOLO


def process(args):
    """Export process"""
    model = YOLO(args.model_od)

    if args.precision_od == "FP32":
        half = False
        int8 = False

    elif args.precision_od == "FP16":
        half = True
        int8 = False
    elif args.precision_od == "INT8":
        half = False
        int8 = True

    data = args.data if args.data else Path(__file__).parent / "VOC2007.yaml"

    model.export(
        format=args.format,
        device=args.device,
        half=half,
        int8=int8,
        imgsz=args.img_size,
        batch=args.batch_size,
        opset=args.opset,
        workspace=args.workspace,
        data=data,
    )
