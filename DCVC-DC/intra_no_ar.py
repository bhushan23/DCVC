from src.models.image_model import IntraNoAR, IntraEncoder
from src.utils.stream_helper import get_state_dict
from tetrai import coremltools_extensions as cte

import coremltools as ct
import numpy as np
import tetra_hub as hub
import torch
import torch.nn as nn
import skimage


def validate(torch_outputs, coreml_outputs, output_names, psnr_threshold=60):
    for A, B, name in zip(torch_outputs, coreml_outputs, output_names):
        data_range = max(A.max() - A.min(), np.abs(A).max())
        if (A == B).all():
            psnr = np.inf
        else:
            psnr = skimage.metrics.peak_signal_noise_ratio(A, B, data_range=data_range)

        if psnr < psnr_threshold:
            print(f"PSNR drop for {name}: {psnr} < threshold ({psnr_threshold}).\n" + f"Comparing: {A} \n {B}")

def _update_torch_outputs(outputs):
    new_outputs = []
    for each in outputs:
        out = each.detach().numpy()
        if out.shape == ():
            out = np.array([out])
        new_outputs.append(out)
    return new_outputs

class IntraNoAR_wrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = IntraNoAR()
        state_dict = get_state_dict(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, x):
        out_dict = self.model(x)
        return out_dict["x_hat"], out_dict["bit"], out_dict["bpp"], out_dict["bpp_y"], out_dict["bpp_z"]

class IntraNoAR_encoder_wrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = IntraNoAR()
        state_dict = get_state_dict(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, x):
        out_dict = self.model(x)
        return out_dict["x_hat"], out_dict["bit"], out_dict["bpp"], out_dict["bpp_y"], out_dict["bpp_z"]

model_path = '/Volumes/Common/work/customer/ms/dcvc/DCVC_main/DCVC-DC/checkpoints/cvpr2023_image_psnr.pth.tar'
model = IntraNoAR_wrapper(model_path)
model.eval()

"""
model = IntraNoAR()
model_path = '/Volumes/Common/work/customer/ms/dcvc/DCVC/DCVC-DC/checkpoints/cvpr2023_image_psnr.pth.tar'
state_dict = get_state_dict(model_path)
model.load_state_dict(state_dict)
model.eval()
"""

shape_map = {
    "1080p" : (1, 3, 1088, 1920),
    "720p" : (1, 3, 768, 1280),
    "360p" : (1, 3, 384, 512)
}


for model_name in ["360p"]: #, "720p", "1080p"]:
    input_shape = shape_map[model_name]

    model_path = "tetra/models/"
    model_name = model_path + "IntraNoAr_forward_" + model_name + ".mlmodel"

#    x = torch.ones(input_shape)
#    traced_model = torch.jit.trace(model, x, check_trace=False, strict=False)
    # print(traced_model.graph)

    # cml_inputs = [ ct.TensorType('x', shape=x.shape) ]
    # mlmodel = cte.convert(traced_model, convert_to="neuralnetwork", inputs=cml_inputs)
    # mlmodel.save(model_name)
    mlmodel = ct.models.MLModel('tetra/models/IntraNoAr_forward_360p.mlmodel') #, compute_units=ct.ComputeUnit.CPU_ONLY)


#    sample = torch.randint(1, 3, input_shape).float()
    sample = torch.ones(input_shape).float()
#    sample = torch.randn(input_shape)
    torch_outputs = model(sample)
    torch_outputs = _update_torch_outputs(torch_outputs)
    print(torch_outputs)

    inputs = { "x" : sample.numpy().astype(np.float32) }
    coreml_output = mlmodel.predict(inputs)
    print(coreml_output)

    torch_output_order = ["var_2314", "var_2452", "var_2453", "bpp_y", "bpp_z"]
    coreml_output_values = [coreml_output[key] for key in torch_output_order]
    validate(torch_outputs, coreml_output_values)

    exit(0)
    devices = [hub.Device(name='Apple iPhone 14 Pro')] #  , hub.Device(name='Apple iPhone 12 Pro', os='16.2')] #, hub.Device(name='Apple iPhone 13 Pro', os='15.2')]
    for each_device in devices:

        """
        hub.submit_profile_job(
            model=model_name,
            name=model_name,
            # input_shapes={ 'x', x.shape },
            device=each,
            options="--apple_zero_copy" # --quantize_weight num_bits=8,method=linear_symmetric"
        )
        """

        validation_job = hub.submit_validation_job(
                model=mlmodel,
                device=each_device,
                inputs=inputs,
                options="--apple_zero_copy"
            )

        coreml_output = validation_job.download_output_data()
        print(coreml_output.items())
