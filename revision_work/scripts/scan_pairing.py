"""Explicit two-stream indexing reference. Not installed into historical models."""
import torch


def pack(main, supplement, order):
    if main.shape != supplement.shape or main.ndim != 3:
        raise ValueError('Expected equal (batch, channel, length) inputs')
    if order == 'interleaved':
        return torch.stack((main, supplement), dim=-1).flatten(-2)
    if order == 'blocked':
        return torch.cat((main, supplement), dim=-1)
    raise ValueError(order)


def restore(sequence, length, order, reduction='mean'):
    if sequence.shape[-1] != 2*length:
        raise ValueError('The two streams must both be retained')
    if order == 'interleaved':
        first, second = sequence[..., ::2], sequence[..., 1::2]
    elif order == 'blocked':
        first, second = sequence[..., :length], sequence[..., length:]
    else:
        raise ValueError(order)
    if reduction == 'sum':
        return first + second
    if reduction == 'mean':
        return (first + second)/2
    raise ValueError(reduction)
