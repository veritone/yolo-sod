import json
import subprocess
import torch


def get_video_properties(file_path: str):
    """Get the width, height, fps and codec of the video file using ffprobe."""
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            file_path,
        ],
        capture_output=True,
    )
    file_info = json.loads(process.stdout.decode())
    width = file_info["streams"][0]["width"]
    height = file_info["streams"][0]["height"]
    fps = file_info["streams"][0]["avg_frame_rate"]
    fps = int(fps.split("/")[0]) / int(fps.split("/")[1])
    return {
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "codec": file_info["streams"][0]["codec_name"],
    }


def yuv_to_rgb(frames):
    """Convert YUV frames to RGB in GPU."""
    frames = frames.to(torch.float32)
    y = frames[..., 0, :, :]
    u = frames[..., 1, :, :]
    v = frames[..., 2, :, :]

    y /= 255
    u = u / 255 - 0.5
    v = v / 255 - 0.5

    r = y + 1.14 * v
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u

    rgb = torch.stack([r, g, b], 1)
    rgb.clamp_(0, 1.0)
    return rgb


class JSON_Writer:
    """Write the output of the OD model to a JSON file."""

    def __init__(
        self,
        file_path: str,
        class_names: dict,
        width_scale: float,
        height_scale: float,
        fps: int,
    ):
        self.file_path = file_path
        self.class_names = class_names
        self.width_scale = width_scale
        self.height_scale = height_scale
        self.fps = fps

        self.output = {"label": {}}
        self.idx = 0

    def parse_frame(self, frame):
        """Parse the output of the OD model and write it to the JSON file."""
        timestamp = f"{self.idx * 1000.0 / self.fps}"
        self.output["label"][timestamp] = []
        for obj in frame.boxes:
            category = self.class_names[int(obj.cls.cpu().numpy()[0])]
            obj_data = []
            xyxy = obj.xyxy.cpu().numpy().tolist()[0]
            xyxy[0] *= self.width_scale
            xyxy[1] *= self.height_scale
            xyxy[2] *= self.width_scale
            xyxy[3] *= self.height_scale
            obj_data.append(xyxy)
            obj_data.append(category)
            obj_data.append(float(obj.conf.cpu().numpy()[0]))

            self.output["label"][timestamp].append(obj_data)
        self.idx += 1

    def save(self):
        """Save the JSON output to the file."""
        with open(self.file_path, "w") as f:
            json.dump(self.output, f)


def get_scale_factor(width, height, imgsz):
    """Calculate the scale factor for the input image."""
    width_aspects = imgsz[1] / width
    height_aspects = imgsz[0] / height
    aspect_ratio = min(width_aspects, height_aspects)
    # Ensure that the width and height are divisible by 2 for NVDEC
    decode_width = int(width * aspect_ratio / 2) * 2
    decode_height = int(height * aspect_ratio / 2) * 2
    return 1 / aspect_ratio, decode_width, decode_height
