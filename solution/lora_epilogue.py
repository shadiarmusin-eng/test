"""Fused LoRA epilogue: out = base + lora * scale as ONE Triton kernel (written from scratch).

PEFT computes `result + lora_B(lora_A(dropout(x))) * scaling` with separate multiply, add and
cast kernels. Here the same math runs in one kernel (fp32 inside, rounded once to the output
dtype). Dropout, A and B are called exactly as PEFT calls them, so the RNG stream is unchanged.
Only the plain case is handled; everything else goes to PEFT's original forward.
"""
import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _epi_fwd(R, L, Y, n, scale, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    r = tl.load(R + offs, mask=m, other=0.0).to(tl.float32)
    l = tl.load(L + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + offs, (r + l * scale).to(Y.dtype.element_ty), mask=m)


@triton.jit
def _epi_bwd(G, D, n, scale, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(D + offs, (g * scale).to(D.dtype.element_ty), mask=m)


class LoraEpilogue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, base, lora, scale):
        base = base.contiguous()
        lora = lora.contiguous()
        assert base.shape == lora.shape
        out = torch.empty_like(base)
        n = base.numel()
        _epi_fwd[(triton.cdiv(n, _BLOCK),)](base, lora, out, n, scale,
                                            BLOCK=_BLOCK, num_warps=4)
        ctx.scale = scale
        ctx.lora_dtype = lora.dtype
        return out

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous()
        dl = torch.empty(g.shape, dtype=ctx.lora_dtype, device=g.device)
        n = g.numel()
        _epi_bwd[(triton.cdiv(n, _BLOCK),)](g, dl, n, ctx.scale, BLOCK=_BLOCK, num_warps=4)
        return g, dl, None


def patch_lora_epilogue():
    from peft.tuners.lora import layer as lora_layer

    cls = lora_layer.Linear
    if getattr(cls, "_soup_epilogue", False):
        return
    orig = cls.forward

    def forward(self, x, *args, **kwargs):
        name = self.active_adapters[0] if len(self.active_adapters) == 1 else None
        plain = (name is not None and not args and not kwargs and x.is_cuda
                 and not self.disable_adapters and not self.merged
                 and name in self.lora_A
                 and not getattr(self, "use_dora", {}).get(name, False)
                 and getattr(self, "lora_variant", {}).get(name) is None)
        if not plain:
            return orig(self, x, *args, **kwargs)
        result = self.base_layer(x)
        out_dtype = result.dtype
        lora_a, lora_b = self.lora_A[name], self.lora_B[name]
        xin = self._cast_input_dtype(x, lora_a.weight.dtype)
        lora = lora_b(lora_a(self.lora_dropout[name](xin)))
        out = LoraEpilogue.apply(result, lora, float(self.scaling[name]))
        return out.to(out_dtype)

    cls.forward = forward
    cls._soup_epilogue = True
