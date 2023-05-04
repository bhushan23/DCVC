import os
import argparse
import coremltools as ct
import torch
import torchvision.transforms as T
import tetra_hub as hub

from src.transforms.functional import ycbcr420_to_444
from src.utils.stream_helper import get_padding_size
from src.utils.video_reader import YUVReader
from src.tetra.utils import *
from src.tetra.model_wrapper import *

NUM_OF_FRAMES_PER_VALIDATION_JOB = 1000

def _np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image

"""
Structure:
    intra_no_ar.py -> runs inference on pytorch model locally and runs validation job for respective mlmodel
    utils.py -> utility methods for validation
    model_wrapper.py -> wrapper over encoder, decoder for IntraNoAR model
"""

current_dir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--profile_job_id", type=str, default="ygzr2658")
parser.add_argument("--yuv_path", type=str, required=True)
parser.add_argument("--yuv_frame_count", type=int, default=1)
parser.add_argument("--yuv_ht", type=int, required=True)
parser.add_argument("--yuv_wt", type=int, required=True)

args = parser.parse_args()

input_shape = (1, 3, 360, 480)
model_name = f"IntraNoAR_forward_360.mlmodel"

# Device to run validation on
device = hub.Device(name="Apple iPhone 14 Pro")

# Get existing profile job
job = hub.get_job(args.profile_job_id)
# Get target model from profiling job to validate data on
target_model = job.get_target_model()

image_dims = tuple(input_shape[2:])
src_reader = YUVReader(args.yuv_path, args.yuv_wt, args.yuv_ht)
tranform = T.Resize(image_dims)

input_frames = []
for frame_num in range(args.yuv_frame_count):
    y, uv = src_reader.read_one_frame(dst_format="420")
    yuv = ycbcr420_to_444(y, uv, order=0)
    x = _np_image_to_tensor(yuv)
    x = tranform(x)

    # pad if necessary
    padding_l, padding_r, padding_t, padding_b = get_padding_size(*image_dims, 16)
    sample = torch.nn.functional.pad(
        x,
        (padding_l, padding_r, padding_t, padding_b),
        mode="replicate",
    )

    input_frames.append(sample)


for i in range(len(input_frames) // NUM_OF_FRAMES_PER_VALIDATION_JOB + 1):
    index = NUM_OF_FRAMES_PER_VALIDATION_JOB * i
    sample_frames = input_frames[ index : index + NUM_OF_FRAMES_PER_VALIDATION_JOB ]
    sample_frames = [ sample.numpy().astype(np.float32) for sample in sample_frames ]

    # 3. submit validation job
    print(f"Running validation job for {len(sample_frames)} frames({index} - {index + len(sample_frames)}).")
    inputs = { "x" : sample_frames }
    validation_job = hub.submit_validation_job(
        model=target_model,
        name=model_name,
        device=device,
        inputs=inputs,
    )

    # 4. Collect output from validation job
    coreml_output = validation_job.download_output_data()
    if coreml_output is None:
        print("Validation failed! Please try running on the same device again or new device.")
        exit(0)

    torch_output_order = list(coreml_output.keys())
    coreml_output_values = [coreml_output[key] for key in torch_output_order][0]

    print('Performing PSNR check')
    # PSNR check: coreml vs input
    print('--| PSNR check coreml out wrt input |--')
    frame_names = [ f'frame_{i}' for i in range(len(sample_frames))]
    validate(sample_frames, coreml_output_values, frame_names, psnr_threshold=0.)
