import torch
from ultralytics import YOLO
from yolo_sod.utils import JSON_Writer, yuv_to_rgb, get_scale_factor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, PILToTensor
from PIL import Image
from pathlib import Path


class ImageDataset(Dataset):
    """Dataset class for processing a directory of images"""

    def __init__(self, directory, imgsz):
        """Initialize the dataset

        Can be used with a DataLoader to return batches of images. Each image is
        resized to the requested imgsz, preserving the aspect ratio and using
        zero padding. The scale factor is returned with each image, so predicted
        bounding boxes can be rescaled to the original image space.

        Args:
            directory (str): Path to the directory containing the images
            imgsz (Tuple[int, int]): Size of the images to be processed
        """
        self.directory = directory
        self.image_files = sorted(path for path in Path(directory).glob("*"))
        self.pil_to_tensor = PILToTensor()
        self.imgsz = imgsz

    def __len__(self):
        """Return the number of images in the dataset"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Return the image and scale factor at the given index. Images are of
        YUV format."""
        image = Image.open(self.image_files[idx]).convert("YCbCr")
        scale_factor, decode_width, decode_height = get_scale_factor(
            image.size[0], image.size[1], self.imgsz
        )
        padding = (0, self.imgsz[1] - decode_width, 0, self.imgsz[0] - decode_height)
        transforms = [
            self.pil_to_tensor,
            Resize((decode_height, decode_width)),
            torch.nn.ZeroPad2d(padding),
        ]
        for transform in transforms:
            image = transform(image)
        return image.to(0), scale_factor


def process(args):
    model = YOLO(args.model_od, task="detect")

    class_names = model.names
    imgsz = model.predictor.model.imgsz
    batch_size = model.predictor.model.batch
    fps = 1  # Image reader treats each image as a frame in a 1 FPS video
    width_scale = 1
    height_scale = 1

    json_writer = JSON_Writer(args.output, class_names, width_scale, height_scale, fps)
    # Images begin at index 1
    json_writer.idx = 1
    image_dataset = DataLoader(ImageDataset(args.input, imgsz), batch_size=batch_size)

    for image in image_dataset:
        rgb = yuv_to_rgb(image[0])
        if rgb.shape[0] < batch_size:
            rgb = torch.cat(
                [
                    rgb,
                    torch.zeros(
                        batch_size - rgb.shape[0],
                        *rgb.shape[1:],
                        device=rgb.device,
                    ),
                ],
                dim=0,
            )
        results = model.predict(
            source=rgb,
            stream=True,
            classes=[
                idx
                for idx, category in class_names.items()
                if category in args.object_categories
            ],
            imgsz=imgsz,
            batch=batch_size,
            conf=args.confidence_threshold_od,
            iou=args.nms_iou_threshold,
        )
        for frame, scaling_factor in zip(results, image[1]):
            json_writer.width_scale = float(scaling_factor)
            json_writer.height_scale = float(scaling_factor)
            json_writer.parse_frame(frame)
    json_writer.save()
