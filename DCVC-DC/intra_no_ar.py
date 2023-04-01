import os
import argparse
import coremltools as ct
import torch
import tetra_hub as hub

from utils import *
from model_wrapper import *

"""
Structure:
    intra_no_ar.py -> runs inference on pytorch model locally and runs validation job for respective mlmodel
    utils.py -> utility methods for validation
    model_wrapper.py -> wrapper over encoder, decoder for IntraNoAR model
"""

current_dir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=os.path.join(current_dir, "checkpoints", "cvpr2023_image_psnr.pth.tar"))
parser.add_argument("--model_type", type=str, default="encoder") # pass either "encoder" or "decoder"
parser.add_argument("--model_size_to_test", type=str, default="360p") # pass input size to work with
args = parser.parse_args()

model_path = args.model_path
model_type = args.model_type
model_size_to_test = args.model_size_to_test

# initialize wrapper model
# model = IntraNoAR_wrapper(model_path, mode=mode)
if model_type == "encoder":
    model = IntraNoAR_encoder_wrapper(model_path)
elif model_type == "decoder":
    model = IntraNoAR_decoder_wrapper(model_path)
else:
    raise RuntimeError("Unsupported model_type '{model_type}' specified.")
model.eval()

# NOTE: we have included padding for simplicity
shape_map = {
    "1080p" : (1, 3, 1088, 1920),
    "720p" : (1, 3, 768, 1280),
    "360p" : (1, 3, 384, 512)
}

if model_size_to_test not in shape_map:
    raise RuntimeError("fPlease provide correct model_size_to_test. Provided {model_size_to_test}.")

input_shape = shape_map[model_size_to_test]

if model.mode == "decoder":
    # TODO: check decoder input size observation with MS
    # Decoder shape is following
    # [1, 256, inputH // 16, inputW // 16]
    input_shape = [input_shape[0], 256, input_shape[2] // 16, input_shape[3] // 16]

model_name = f"IntraNoAR_{model.mode}_{model_size_to_test}.mlmodel"
model_path = os.path.join(current_dir, "tetra", "models", model_name)

# TODO: replace by helper routine
# sample = _image_to_torch(input_shape)
sample = torch.rand(input_shape)

# PyTorch inference: expected result.
torch_outputs = model(sample)
torch_outputs = update_torch_outputs(torch_outputs)

# CoreML inference: observed result.
inputs = { "x" : [ sample.numpy().astype(np.float32) ]}

# device = hub.Device(name="Apple iPhone 14 Pro")
# device = hub.Device(name="Apple iPhone 13")
device = hub.Device(name='Apple iPhone 13 Pro Max', os='15.1')
"""
TODO: Uncomment this on April 11th 2023 i.e. once clip, const-elimination fixes are released.

x = torch.ones(input_shape)
traced_model = torch.jit.trace(model, x, check_trace=False, strict=False)

job = hub.submit_profile_job(
    model=mlmodel,
    name=model_name,
    input_shapes={ "x", input_shape  },
    device=device,
    options="--apple_zero_copy"
)

mlmodel = job.download_target_model()
"""
validation_job = hub.submit_validation_job(
        model=model_path,
        name=model_name,
        device=device,
        inputs=inputs,
        options="--apple_zero_copy"
    )

coreml_output = validation_job.download_output_data()
if coreml_output is None:
    print("Validation failed! Please try running on the same device again or new device.")
    exit(0)

torch_output_order = list(coreml_output.keys()) if model.torch_output_order is None else model.torch_output_order
coreml_output_values = [coreml_output[key][0] for key in torch_output_order]
validate(torch_outputs, coreml_output_values, torch_output_order)
