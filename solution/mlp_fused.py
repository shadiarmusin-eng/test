"""Qwen2 MLP tail: silu(gate)*up and the 4-bit down projection as ONE autograd node.

The frozen down_proj has no weight gradient, so its input `silu(gate)*up` does not have to be
kept for the backward pass: only `gate` and `up` are saved (and the product is never stored).
The backward needs just  d_s = d_y @ W  (W de-quantised by our NF4 kernel) and the SwiGLU
backward. Result: one fp16 activation tensor less per layer, one autograd node less per MLP.

Idea source: "save fewer activations when a weight is frozen" (standard autograd reasoning);
the kernels used are the example's SwiGLU launches and our NF4 de-quantiser. Own code.
Anything unusual (LoRA on down_proj, other quant type, no grad, dtype mismatch) goes the old way.
"""
import torch
import triton

from .swiglu import BLOCK, example_swiglu_bwd_kernel, example_swiglu_fwd_kernel, swiglu


def _deq(wq, qs):
    import bitsandbytes.functional as BF
    return BF.dequantize_4bit(wq, qs)  # [out, in] in qs.dtype (our kernel once patched)


class MLPTail(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u, wq, qs):
        g = g.contiguous()
        u = u.contiguous()
        s = torch.empty_like(g)
        n = g.numel()
        example_swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](g, u, s, n, BLOCK=BLOCK)
        y = s @ _deq(wq, qs).t()
        ctx.save_for_backward(g, u)
        ctx.wq, ctx.qs = wq, qs
        return y

    @staticmethod
    def backward(ctx, dy):
        g, u = ctx.saved_tensors
        w = _deq(ctx.wq, ctx.qs)
        ds = (dy.to(w.dtype) @ w).contiguous()
        dg = torch.empty_like(g)
        du = torch.empty_like(u)
        n = g.numel()
        example_swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](ds, g, u, dg, du, n, BLOCK=BLOCK)
        return dg, du, None, None


def patch_mlp_tail():
    from transformers.models.qwen2 import modeling_qwen2 as Q

    cls = Q.Qwen2MLP
    if getattr(cls, "_soup_tail", False):
        return
    prev = cls.forward

    def forward(self, x):
        dp = self.down_proj
        w = getattr(dp, "weight", None)
        qs = getattr(w, "quant_state", None)
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if (act != "silu" or not x.is_cuda or qs is None or hasattr(dp, "lora_A")
                or getattr(qs, "quant_type", None) != "nf4"
                or getattr(dp, "bias", None) is not None
                or not torch.is_grad_enabled()):
            return prev(self, x)
        g = self.gate_proj(x)
        u = self.up_proj(x)
        if g.dtype != qs.dtype or u.dtype != g.dtype:
            return dp(swiglu(g, u))
        return MLPTail.apply(g, u, w.data, qs)

    cls.forward = forward
    cls._soup_tail = True
