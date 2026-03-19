import math
import torch

device = torch.device('cuda')
# device = torch.device('cpu')

from rDL_model.common import generate_pattern

PADDING_REPEAT = 16


def denoise2DSIM_model_inference(model, data):
    devicein = data.device
    assert data.shape[0] == 1
    # if data.shape[-1] == 3072:
    #     border = 256
    #     result = torch.zeros_like(data)
    #     result[..., :1536, :1536] = model(data[..., : 1536 + border, : 1536 + border].to(device))[..., :1536, :1536].to(devicein)
    #     result[..., :1536, -1536:] = model(data[..., : 1536 + border, -1536 - border:].to(device))[..., :1536, -1536:].to(devicein)
    #     result[..., -1536:, :1536] = model(data[..., -1536 - border:, : 1536 + border].to(device))[..., -1536:, :1536].to(devicein)
    #     result[..., -1536:, -1536:] = model(data[..., -1536 - border:, -1536 - border:].to(device))[..., -1536:, -1536:].to(devicein)
    #     return result
    # else:
    #     return model(data.to(device)).to(devicein)
    return model(data.to(device)).to(devicein)


def reconstruction2DSIM_model_inference(model, data, para, json, half=False, minipatch=1024):
    devicein = data.device
    assert data.shape[0] == 1

    _, C, H, W = data.shape

    # praw = generate_pattern(data, para, json, pattern_size='raw')
    psim = generate_pattern(data, para, json, pattern_size='sim')

    if data.shape[-1] <= minipatch:
        if hasattr(model, 'forward_with_pattern'):
            result = model.forward_with_pattern(data.cuda(), None, psim.cuda()).to(devicein)
        else:
            result = model.forward(data.cuda()).to(devicein)

    else:
        result = torch.zeros((1, 1, 2 * H, 2 * W), dtype=data.dtype, device=data.device)
        for h in list(range(0, data.shape[-1], minipatch)):
            for w in list(range(0, data.shape[-1], minipatch)):

                uppad = PADDING_REPEAT if h > 0 else 0
                downpad = PADDING_REPEAT if h < data.shape[-1] else 0
                leftpad = PADDING_REPEAT if w > 0 else 0
                rightpad = PADDING_REPEAT if w < data.shape[-1] else 0

                if hasattr(model, 'forward_with_pattern'):
                    result[..., 2 * h:2 * h + minipatch * 2, 2 * w:2 * w + minipatch * 2] = model.forward_with_pattern(
                        data[..., h - uppad:h + downpad + minipatch, w - leftpad:w + rightpad + minipatch].cuda(),
                        None,
                        psim[..., 2 * h - 2 * uppad:2 * h + 2 * downpad + minipatch * 2, 2 * w - 2 * leftpad:2 * w + 2 * rightpad + minipatch * 2].cuda()
                    )[..., 2 * uppad:2 * uppad + minipatch * 2, 2 * leftpad:2 * leftpad + minipatch * 2].to(devicein)

                else:
                    result[..., 2 * h:2 * h + minipatch * 2, 2 * w:2 * w + minipatch * 2] = model.forward(
                        data[..., h - uppad:h + downpad + minipatch, w - leftpad:w + rightpad + minipatch].cuda(),
                    )[..., 2 * uppad:2 * uppad + minipatch * 2, 2 * leftpad:2 * leftpad + minipatch * 2].to(devicein)

    return result
