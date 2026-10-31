import pytest
import torch
import torch.nn.functional as F

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_mlp_tail():
    import bitsandbytes.functional as BF
    from solution.mlp_fused import MLPTail
    torch.manual_seed(0)
    W = torch.randn(256, 1024, device="cuda", dtype=torch.float16) * 0.05
    q, st = BF.quantize_4bit(W, blocksize=64, compress_statistics=True, quant_type="nf4")
    Wd = BF.dequantize_4bit(q, st)
    g = torch.randn(3, 9, 1024, device="cuda", dtype=torch.float16).requires_grad_()
    u = torch.randn(3, 9, 1024, device="cuda", dtype=torch.float16).requires_grad_()
    g2, u2 = g.detach().clone().requires_grad_(), u.detach().clone().requires_grad_()
    y = MLPTail.apply(g, u, q, st)
    y_ref = (F.silu(g2.float()) * u2.float()).half() @ Wd.t()
    torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=2e-2)
    dy = torch.randn_like(y)
    y.backward(dy)
    y_ref.backward(dy)
    torch.testing.assert_close(g.grad, g2.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(u.grad, u2.grad, rtol=3e-2, atol=3e-2)
