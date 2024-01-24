from src.tetra.model_wrapper import *

import argparse
import os
import torch

from coremltools_extensions import convert as cte_convert
import coremltools as ct

def convert_with_extesions(traced_model, model_name, output_dir, input_shapes, convert_to="mlpackage", os_version="16.2"):
    """
        Converts traced torch model into coreml model format
        traced_model: Traced torch model
        model_name: name of the converted model to set during serialization (without extensions)
        output_dir: output dir to save model to
        input_shapes: List of tuple of input shapes in order
        convert_to: Convert to "mlpackage" or "neuralnetwork" format of CoreML. Default: "mlpackage"
        os_version: target OS version. Setting this to 13, 14, 15, 16 triggers related graph passes
            during conversion. Default: "16.2"
    """
    ct_inputs = [ ct.TensorType(shape=shape) for shape in input_shapes ]
    model = cte_convert(traced_model, inputs=ct_inputs, os_version=os_version)
    output_path = os.path.join(output_dir, model_name)
    model_output_path = f"{output_path}.mlpackage"
    model.save(model_output_path)
    print(f"Converted model saved to {model_output_path}.")


# Example usage:
# python convert_e2e_models.py --i_frame_model_path ./checkpoints/cvpr2023_image_yuv420_psnr.pth.tar --p_frame_model_path ./checkpoints/cvpr2023_video_yuv420_psnr.pth.tar --output_assets_dir ./converted_e2e_assets
#
parser = argparse.ArgumentParser(description="Converting DCVC models to CoreML model format")
parser.add_argument(
    "--output_assets_dir", type=str, required=True
)
parser.add_argument(
    "--i_frame_model_path", type=str, required=True
)
parser.add_argument(
    "--p_frame_model_path", type=str, required=True
)
parser.add_argument(
    "--model_input_size", type=str, default="360p", choices=["360p", "720p", "1080p"]
)
args = parser.parse_args()

asset_dir = args.output_assets_dir
image_model_path = args.i_frame_model_path
video_model_path = args.p_frame_model_path
model_input_size = args.model_input_size

os.makedirs(asset_dir, exist_ok=True)

# Load IntraNoAR model wrappers
i_frame_net = IntraNoAR_wrapper(model_path=image_model_path)
i_frame_net.eval()


# Load DMC model wrappers
p_frame_net = DMC_wrapper(model_path=video_model_path)
p_frame_net.eval()

# dummy input to ensure q_index is not constant folded
sample_q_index = torch.Tensor([0]).reshape(1,).type(torch.long)
frame_index = torch.Tensor([0]).reshape(1,).type(torch.long)
dummy_input = torch.Tensor([0]).reshape(1,)

# Following input shapes are computed w.r.t 360p model
# input shapes
shape_map = {
    "1080p" : (1, 3, 1088, 1920),
    "720p" : (1, 3, 720, 1280),
    "360p" : (1, 3, 368, 480)
}

x_shape = shape_map[model_input_size]

# trace and convert IntraNoAR model

x = torch.rand(x_shape)
traced_model = torch.jit.trace(i_frame_net, (x, sample_q_index, dummy_input), check_trace=False)
convert_with_extesions(traced_model=traced_model, model_name=f"IntraNoAR_e2e_{model_input_size}", output_dir=asset_dir, input_shapes=[x.shape, sample_q_index.shape, dummy_input.shape])


#
# trace and convert DMC model
#

traced_model = torch.jit.trace(p_frame_net, (x, x, sample_q_index, dummy_input, frame_index), check_trace=False)
convert_with_extesions(traced_model=traced_model, model_name=f"DMC_e2e_{model_input_size}", output_dir=asset_dir, input_shapes=[x.shape, x.shape, sample_q_index.shape, dummy_input.shape, frame_index.shape])
