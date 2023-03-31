from src.models.image_model import IntraNoAR
from src.utils.stream_helper import get_state_dict

import os
import argparse
import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import skimage

import tetra_hub as hub

"""
Helper routines
"""
def validate(torch_outputs, coreml_outputs, output_names, psnr_threshold=60):
    for A, B, name in zip(torch_outputs, coreml_outputs, output_names):
        data_range = max(A.max() - A.min(), np.abs(A).max())
        if (A == B).all():
            psnr = np.inf
        else:
            psnr = skimage.metrics.peak_signal_noise_ratio(A, B, data_range=data_range)

        print(f"PSNR for {name}: {psnr}")
        if psnr < psnr_threshold:
            print(f"PSNR drop for {name}: {psnr} < threshold ({psnr_threshold}).\n" + f"Comparing: {A} \n {B}")

def _update_torch_outputs(outputs):
    new_outputs = []
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    for each in outputs:
        out = each.detach().numpy()
        if out.shape == ():
            out = np.array([out])
        new_outputs.append(out)
    return new_outputs


class IntraNoAR_wrapper(nn.Module):
    def __init__(self, model_path, mode="forward"):
        super().__init__()
        self.model = IntraNoAR()
        state_dict = get_state_dict(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        assert mode in { "forward", "encoder", "decoder"}
        self.mode = mode
        self.torch_output_order = None
        if mode == "forward":
            # Forward has multiple outputs and for numerical accuracy test we need to maintain order.
            # NOTE: variable names will change if mlmodel is updated.
            self.torch_output_order = ["var_2314", "var_2452", "var_2453", "bpp_y", "bpp_z"]

    def forward(self, x, q_in_ckpt=False, q_index=0):
        if self.mode == "forward":
            out_dict = self.model(x)
            return out_dict["x_hat"], out_dict["bit"], out_dict["bpp"], out_dict["bpp_y"], out_dict["bpp_z"]
        elif self.mode == "encoder":
            # TODO: add remaining pieces of encoder
            # Reference: image_model.py::IntraNoAR::forward
            curr_q_enc, _ = self.model.get_q_for_inference(q_in_ckpt=q_in_ckpt, q_index=q_index)
            y = self.model.enc(x, curr_q_enc)
            # NOTE: Skipping pad for simplicity.
            # e.g. if input is 720:
            # y is of shape (1, 256, 45, 80)
            # padding is (0, tensor(0), 0, tensor(-3))
            # y_pad becomes torch.Size([1, 256, 48, 80])
            # We need to store how much was padded for decoder
            # to get back to original resolution
            # y_pad, _ = self.model.pad_for_y(y)
            z = self.model.hyper_enc(y)
            z_hat = self.model.quant(z)
            return z_hat
        elif self.mode == "decoder":
            # TODO: add remaining pieces of decoder
            _, curr_q_dec = self.model.get_q_for_inference(q_in_ckpt=q_in_ckpt, q_index=q_index)
            y_hat = self.model.hyper_dec(x)
            params = self.model.y_prior_fusion(y_hat)
            # NOTE: refer to notes from encoder for Skipping pad for simplicity
            # params = self.model.slice_to_y(params, slice_shape)
            # TODO: Update and use decompression correctly.
            # _, y_q, y_hat, scales_hat = self.model.forward_four_part_prior(
            #     y_hat, params, self.model.y_spatial_prior_adaptor_1, self.model.y_spatial_prior_adaptor_2,
            #     self.model.y_spatial_prior_adaptor_3, self.model.y_spatial_prior)
            # y_hat = self.model.decompress_four_part_prior_with_y(y_hat, y_hat,
            #                                         self.model.y_spatial_prior_adaptor_1,
            #                                         self.model.y_spatial_prior_adaptor_2,
            #                                         self.model.y_spatial_prior_adaptor_3,
            #                                         self.model.y_spatial_prior)
            y_hat = self.model.dec(y_hat, curr_q_dec)
            # TODO: UNet model i.e. refine leads to pytorch crash
            # y_hat = self.model.refine(y_hat)
            # y_hat = y_hat.clamp_(0, 1)
            return y_hat


current_dir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default=os.path.join(current_dir, "checkpoints", "cvpr2023_image_psnr.pth.tar"))
parser.add_argument("--model_type", type=str, default="encoder")
parser.add_argument("--model_sizes_to_test", type=str, default="720p") # comma separated list e.g. "360p, 720p, 1080p"
args = parser.parse_args()

model_path = args.model_path
mode = args.model_type
model_sizes_to_test = args.model_sizes_to_test.split(",")

# initialize wrapper model
model = IntraNoAR_wrapper(model_path, mode=mode)
model.eval()

# NOTE: we have included padding for simplicity
shape_map = {
    "1080p" : (1, 3, 1088, 1920),
    "720p" : (1, 3, 768, 1280),
    "360p" : (1, 3, 384, 512)
}

for model_name in model_sizes_to_test:
    input_shape = shape_map[model_name]
    if model.mode == "decoder":
        # TODO: check decoder input size observation with MS
        # Decoder shape is following
        # [1, 256, inputH // 16, inputW // 16]
        input_shape = [input_shape[0], 256, input_shape[2] // 16, input_shape[3] // 16]

    model_name = f"IntraNoAR_{model.mode}_{model_name}.mlmodel"
    model_path = os.path.join(current_dir, "tetra", "models", model_name)

    # x = torch.ones(input_shape)
    # traced_model = torch.jit.trace(model, x, check_trace=False, strict=False)

    # TODO: replace by helper routine
    # sample = _image_to_torch(input_shape)
    sample = torch.rand(input_shape)

    # PyTorch inference: expected result.
    torch_outputs = model(sample)
    torch_outputs = _update_torch_outputs(torch_outputs)

    # CoreML inference: observed result.
    inputs = { "x" : [ sample.numpy().astype(np.float32) ]}
    mlmodel = ct.models.MLModel(model_path) #, compute_units=ct.ComputeUnit.CPU_ONLY)

    devices = [hub.Device(name="Apple iPhone 14 Pro")]
    for each_device in devices:
        """
        Once we have fixes into hub, we can start using this
        job = hub.submit_profile_job(
            model=model_name,
            name=model_name,
            # input_shapes={ "x", x.shape },
            device=each,
            options="--apple_zero_copy"
        )

        mlmodel = job.download_target_model()
        """

        validation_job = hub.submit_validation_job(
                model=model_path,
                name=model_name,
                device=each_device,
                inputs=inputs,
                options="--apple_zero_copy"
            )

        coreml_output = validation_job.download_output_data()
        torch_output_order = list(coreml_output.keys()) if model.torch_output_order is None else model.torch_output_order
        coreml_output_values = [coreml_output[key][0] for key in torch_output_order]
        validate(torch_outputs, coreml_output_values, torch_output_order)
