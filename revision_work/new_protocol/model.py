# Adapted from the local Ultralytics project, AGPL-3.0.
from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

from ultralytics.nn.tasks import DetectionModel, parse_model
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.timeguid import TimeMambaStem, TMixSSM
from ultralytics.nn.modules.tgsmamba import SS2D1conv
from ultralytics.utils.torch_utils import initialize_weights


class ExplicitSTF(TimeMambaStem):
    """Same active computation, with alignment errors surfaced instead of swallowed."""
    def forward(self, x):
        x1 = self.spatial_dw1(x)
        x2 = self.spatial_dw2(x1)
        x3 = self.spatial_dw3(x2)
        x4 = self.spatial_dw4(x3)
        frames = [torch.stack([x1[:, i], x2[:, i], x3[:, i], x4[:, i]], dim=1)
                  for i in range(self.n_frames)]
        reference = frames[-1]
        for i in range(self.n_frames - 1):
            frames[i] = self.alignment(frames[i], reference)
        frames[-1] = self.t4_proj(reference)
        temporal = torch.cat(frames, dim=1)
        y = self.mamba(temporal) + temporal[:, -1:]
        return self.act(self.bn(self.out_proj(y)))


class ExplicitDetect(Detect):
    """SAAM is explicitly active in every approved variant, including E1."""
    def forward(self, x):
        if self.end2end:
            raise ValueError('End-to-end head is outside the approved model definition')
        outputs = []
        for i in range(self.nl):
            feature = self.aligned[i](x[i])
            outputs.append(torch.cat((self.cv2[i](feature), self.cv3[i](feature)), 1))
        if self.training:
            return outputs
        y = self._inference(outputs)
        return y if self.export else (y, outputs)


class SSEModel(DetectionModel):
    def __init__(self, task, device='cpu', verify_stride=False):
        # Avoid the upstream constructor's forced CUDA and hardcoded 640 stride probe.
        nn.Module.__init__(self)
        with open(task['model_yaml']) as f:
            self.yaml = yaml.safe_load(f)
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=8, verbose=False)
        self.names = {0: 'target'}
        self.nc = 1
        self.inplace = True
        self.end2end = False
        self.task = 'detect'
        self.args = SimpleNamespace(cube=True, box=task['loss'][0], cls=task['loss'][1], dfl=task['loss'][2])
        for module in self.model.modules():
            if type(module) is TimeMambaStem:
                module.__class__ = ExplicitSTF
            if type(module) is Detect:
                module.__class__ = ExplicitDetect
            if isinstance(module, TMixSSM) and hasattr(module, 'saam'):
                # This unused member was absent in the matched historical graph.
                del module.saam
            if isinstance(module, SS2D1conv):
                k = task['lfss_kernel']
                module.my_conv = nn.Sequential(*[
                    Conv(module.d_inner, module.d_inner, k=k, g=module.d_inner, p=k // 2)
                    for _ in range(task['lfss_depth'])])
        self.stride = torch.tensor([8., 16., 32.])
        self.model[-1].stride = self.stride
        self.model[-1].inplace = True
        self.model[-1].bias_init()
        initialize_weights(self)
        self.to(device)
        if verify_stride:
            self.model.eval()
            self.model[-1].training = True
            with torch.no_grad():
                output = self(torch.zeros(1, 8, 128, 192, device=device))
            measured_h = [128 / v.shape[-2] for v in output]
            measured_w = [192 / v.shape[-1] for v in output]
            assert measured_h == measured_w == [8., 16., 32.]
            self.train()


def architecture_summary(model):
    return {
        'parameters': sum(p.numel() for p in model.parameters()),
        'state_shapes': {k: list(v.shape) for k, v in model.state_dict().items()},
        'stf': type(model.model[3]).__name__,
        'ifss_layers': [i for i, m in enumerate(model.model) if isinstance(m, TMixSSM)],
        'lfss': [{'path': n, 'depth': len(m.my_conv), 'kernels': [list(c.conv.kernel_size) for c in m.my_conv]}
                 for n, m in model.named_modules() if isinstance(m, SS2D1conv)],
        'saam_head': type(model.model[-1]).__name__,
        'stride': model.stride.tolist(),
    }
