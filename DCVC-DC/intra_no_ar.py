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
parser.add_argument("--model_path", type=str, default=os.path.join(current_dir, "checkpoints", "cvpr2023_image_psnr.pth.tar")) #  "cvpr2023_image_yuv420_psnr.pth.tar"))
parser.add_argument("--model_type", type=str, default="forward") # pass "forward", "encoder" or "decoder"
parser.add_argument("--model_size", type=str, default="360p") # pass input size to work with
parser.add_argument("--yuv_path", type=str, required=True)
parser.add_argument("--yuv_frame_count", type=int, default=1)
parser.add_argument("--yuv_ht", type=int, required=True)
parser.add_argument("--yuv_wt", type=int, required=True)

args = parser.parse_args()

model_path = args.model_path
model_type = args.model_type
model_size_to_test = args.model_size

# initialize wrapper model
# model = IntraNoAR_wrapper(model_path, mode=mode)
# if model_type == "encoder":
#     model = IntraNoAR_encoder_wrapper(model_path)
# elif model_type == "decoder":
#     model = IntraNoAR_decoder_wrapper(model_path)
if model_type == "forward":
    model = IntraNoAR_wrapper(model_path)
else:
    raise RuntimeError("Unsupported model_type '{model_type}' specified.")
model.eval()

# NOTE: we have included padding for simplicity
shape_map = {
    "1080p" : (1, 3, 1080, 1920),
    "720p" : (1, 3, 720, 1280),
    "360p" : (1, 3, 360, 480)

}

if model_size_to_test not in shape_map:
    raise RuntimeError("fPlease provide correct model_size_to_test. Provided {model_size_to_test}.")

input_shape = shape_map[model_size_to_test]

if model.mode == "decoder":
    # TODO: check decoder input size observation with MS
    # Decoder shape is following
    # [1, 256, inputH // 64, inputW // 64]
    input_shape = [input_shape[0], 256, input_shape[2] // 64, input_shape[3] // 64]

model_name = f"IntraNoAR_{model.mode}_{model_size_to_test}.mlmodel"
model_path = os.path.join(current_dir, "src", "tetra", "models", model_name)

device = hub.Device(name="Apple iPhone 14 Pro")
x = torch.rand(input_shape)
# pad if necessary
padding_l, padding_r, padding_t, padding_b = get_padding_size(*input_shape[2:], 16)
x = torch.nn.functional.pad(
    x,
    (padding_l, padding_r, padding_t, padding_b),
    mode="replicate",
)
# 1. Trace Torch model
traced_model = torch.jit.trace(model, x, check_trace=False, strict=False)

# 2. Submit profile job
job = hub.submit_profile_job(
    model=traced_model,
    name=model_name,
    input_shapes={ "x" : x.shape },
    device=device,
)
# or skip running profiling job and instead fetch model from existing job
# job = hub.get_job("w56ne7go")
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
    # validation_job = hub.get_job("1p3evz52")

    # 4. Collect output from validation job
    coreml_output = validation_job.download_output_data()
    if coreml_output is None:
        print("Validation failed! Please try running on the same device again or new device.")
        exit(0)

    # NOTE: query coreml_output to see other results
    # model.torch_output_order can be used to get the order output of of coreml model
    # same as src/tetra/model_wrapper/<model>
    torch_output_order = model.torch_output_order
    # x_hat
    coreml_output_values = coreml_output[torch_output_order[0]]

    # 5. PyTorch inference: expected result.
    torch_outputs = []
    for sample in sample_frames:
        torch_output = model(torch.Tensor(sample))
        torch_outputs += update_torch_outputs(torch_output[0])

    print('Performing PSNR check')
    frame_names = [ f'frame_{i}' for i in range(len(sample_frames))]
    # # PSNR check: torch vs coreml
    # print('--| PSNR check coreml model wrt fp32 torch model |--')
    # validate(torch_outputs, coreml_output_values, torch_output_order, psnr_threshold=0.)

    # PSNR check: torch vs input
    print('--| PSNR check torch out wrt input |--')
    validate(sample_frames, torch_outputs, frame_names, psnr_threshold=0.)

    # PSNR check: coreml vs input
    print('--| PSNR check coreml out wrt input |--')
    validate(sample_frames, coreml_output_values, frame_names, psnr_threshold=0.)
