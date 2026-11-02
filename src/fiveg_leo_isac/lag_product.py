"""Compute the positive-lag product for hybrid integration."""
import torch
import triton
import triton.language as tl


@triton.jit
def product_kernel(z, i1, i2, w1, w2, mask, out, total: tl.constexpr, cols: tl.constexpr, B: tl.constexpr):
    j = tl.program_id(0) * B + tl.arange(0, B)
    q = j // cols
    d = j % cols
    valid = j < total
    use = tl.load(mask + q, valid, other=0) != 0
    p = tl.load(i1 + q, valid, other=0) * cols + d
    m = tl.load(i2 + q, valid, other=0) * cols + d
    a = tl.load(w1 + q, valid, other=0)
    b = tl.load(w2 + q, valid, other=0)
    pr = tl.load(z + 2 * p, valid & use, other=0) * (1 - a) + tl.load(z + 2 * (p + cols), valid & use, other=0) * a
    pi = tl.load(z + 2 * p + 1, valid & use, other=0) * (1 - a) + tl.load(z + 2 * (p + cols) + 1, valid & use, other=0) * a
    mr = tl.load(z + 2 * m, valid & use, other=0) * (1 - b) + tl.load(z + 2 * (m + cols), valid & use, other=0) * b
    mi = tl.load(z + 2 * m + 1, valid & use, other=0) * (1 - b) + tl.load(z + 2 * (m + cols) + 1, valid & use, other=0) * b
    tl.store(out + 2 * j, pr * mr + pi * mi, valid)
    tl.store(out + 2 * j + 1, pi * mr - pr * mi, valid)

def product(z, i1, i2, w1, w2, mask):
    z = z.contiguous()
    shape = (*i1.shape, z.shape[1])
    out = torch.empty(shape, device=z.device, dtype=z.dtype)
    product_kernel[triton.cdiv(out.numel(), 256),](torch.view_as_real(z), i1, i2, w1, w2, mask, torch.view_as_real(out), out.numel(), z.shape[1], 256)
    return out
