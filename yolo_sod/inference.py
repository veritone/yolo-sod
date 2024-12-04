from ultralytics import YOLO
import itertools
import sys
import torch
import torch.multiprocessing as mp
from torchaudio.io import StreamReader
from yolo_sod import CODEC_MAP
from yolo_sod.utils import (
    get_video_properties,
    yuv_to_rgb,
    JSON_Writer,
    get_scale_factor,
)


def filler(
    queue,
    filler_finished,
    inference_finished,
    input_path,
    batch_size,
    codec,
    decode_width,
    decode_height,
    padding,
):
    """Fill the queue with video frames.

    Queue is populated with batches of frames from the video and their
    corresponding batch sizes (as the last batch may be smaller than the
    others). The filler_finished is set when the video is done. This process is
    kept alive until inference is finished otherwise PyTorch may release the
    video frame from memory before we are done with it.

    Args:
        queue (mp.Queue): Queue to hold video frames
        filler_finished (mp.Event): Event to signal when the video is done
        inference_finsihed (mp.Event): Event that signals the end of the inference process
        input_path (str): Path to the input video
        batch_size (int): Number of frames to process at once
        codec (str): Codec to use for decoding the video
        decode_width (int): Width of the decoded video
        decode_height (int): Height of the decoded video
        padding (Tuple[int, int, int, int]): Padding to apply to the video

    """
    s = StreamReader(input_path)
    s.add_video_stream(
        batch_size,
        decoder=CODEC_MAP[codec],
        hw_accel="cuda:0",
        buffer_chunk_size=-1,
        decoder_option={"resize": f"{decode_width}x{decode_height}"},
    )
    padding_function = torch.nn.ZeroPad2d(padding)

    current_batch_size = batch_size

    finished = False
    try:
        while not finished:  # 1 means EOF
            finished = s.fill_buffer() == 1  # 1 means EOF
            (video,) = s.pop_chunks()
            video = yuv_to_rgb(video)
            video = padding_function(video)
            current_batch_size = video.size(0)
            if current_batch_size < batch_size:
                # Pad the last batch with zeros
                video = torch.cat(
                    [
                        video,
                        torch.zeros(
                            batch_size - video.size(0),
                            *video.size()[1:],
                            device=video.device,
                        ),
                    ],
                    dim=0,
                )
            queue.put((video, current_batch_size))
        # Notify inference loop the video is finished
        filler_finished.set()
        # Keep process alive until inference is finished (otherwise memory may be released)
        inference_finished.wait()
    except Exception as e:
        # Catch any exceptions and notify the inference loop (otherwise it will hang indefinitely)
        print(e)
        filler_finished.set()
        queue.put((None, 0))


def process(args):
    """Inference process."""
    video_properties = get_video_properties(args.input)
    width = video_properties["width"]
    height = video_properties["height"]
    fps = video_properties["fps"]
    codec = video_properties["codec"]

    model = YOLO(args.model_od, task="detect")
    # calling model.names will load the model. This must be done before querying
    # the imgsz and batch size
    class_names = model.names
    imgsz = model.predictor.model.imgsz
    batch_size = model.predictor.model.batch
    scale_factor, decode_width, decode_height = get_scale_factor(width, height, imgsz)
    padding = (0, imgsz[1] - decode_width, 0, imgsz[0] - decode_height)
    json_writer = JSON_Writer(args.output, class_names, scale_factor, scale_factor, fps)

    # Spawn the filler process and start filling the queue with frames
    mp.set_start_method("forkserver")
    queue = mp.Queue(args.queue_size)
    filler_finished = mp.Event()
    inference_finished = mp.Event()
    filler_process = mp.Process(
        target=filler,
        args=(
            queue,
            filler_finished,
            inference_finished,
            args.input,
            batch_size,
            codec,
            decode_width,
            decode_height,
            padding,
        ),
    )
    filler_process.start()

    # Process the frames as they become available
    # Loop while the filler process is still running or there are frames left in the queue
    while not filler_finished.is_set() or not queue.empty():
        video, current_batch_size = queue.get()
        if video is None:
            # Filler process encountered an error
            sys.exit("yolo-sod encountered an error while decoding video")
        results = model.predict(
            source=video,
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
            verbose=args.verbose,
        )
        for frame in itertools.islice(results, current_batch_size):
            json_writer.parse_frame(frame)
    # Notify the filler process that inference is finished so the process can terminate
    inference_finished.set()

    json_writer.save()
