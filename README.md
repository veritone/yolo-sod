# yolo_sod

A wrapper around Ultralytics' YOLO library, used for object detection. Results
are output in a custom JSON format.

# Installation

Installation requires `poetry`.

To install `poetry`, use `pipx`:

``` shell
sudo apt install pipx
pipx install poetry
```

To install `yolo-sod`, from this directory run:

``` shell
poetry install 
```

This will install all the requirements in their own virtual environment. To
enter the environment, from this directory run `poetry shell`.

# Exporting a model

To export a model run

``` shell
yolo-sod-export
```

command from withing the `poetry` environment. Without any arguments this will
build an INT8 precision TensorRT engine for the large YOLOv11 model. This will
download the necessary files and use the VOC2007 dataset from `/import/datasets`
for calibration.

Model files will be saved in the current working directory.

Various configuration options can also be set. These are listed with:

``` shell
yolo-sod-export --help
```

Note that by default a 2 GiB workspace is used for TensorRT. This program takes
a large amount of GPU memory during the calibration and building process, so
setting a larger value may lead to memory allocation errors. For the large
YOLOv11 model a workspace of 2GiB requires just under 8GiB of GPU memory to do
the calibration.

# Model inference

To perform inference on a video run

``` shell
yolo-sod --input <path_to_video_file> --output <path_to_json_file> --model-od <path_to_OD_model>
```

from within the `poetry` shell. This will decode the video using Nvidia's NvDec
GPU decoder and perform object detection using the model. The detections will be
written to JSON output of the form:

``` json
{
  "label": {
    "timestamp1": [
      [[x0, y0, x1, y1], "category", confidence],
      [[x0, y0, x1, y1], "category", confidence],
      ...
    ],
    "timestamp2": [
      [[x0, y0, x1, y1], "category", confidence],
      [[x0, y0, x1, y1], "category", confidence],
      ...
    ],
    ...
  }
}
```

Supported video codecs:

- `av1`
- `h264`
- `hevc`
- `mjpeg`
- `mpeg1video`
- `mpeg2video`
- `mpeg4video`
- `vc1`
- `vp8`
- `vp9`

Additional inference options can be found by running

``` shell
yolo-sod --help
```

# Building Packages

To build packages (source and wheel) run

``` shell
poetry build
```

from within this directory. The packages will be produced in the `./dist`
directory.
