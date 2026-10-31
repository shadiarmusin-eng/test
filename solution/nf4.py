"""NF4 de-quantisation kernel in Triton (written from scratch).

bitsandbytes NF4 layout: each byte holds two 4-bit codes (high nibble first); every `blocksize`
elements share one absmax. With double quantisation the absmax values are 8-bit indices into a
256-entry table, with one fp32 scale per 256 blocks.

This kernel handles NBLK quantisation blocks per program: the absmax of a block is computed once
and used for all its bytes (the previous version gathered it for every byte).
Anything unusual falls back to the original bitsandbytes function.
"""
import atexit
import sys
import traceback

import torch
import triton
import triton.language as tl

NBLK = 16  # quantisation blocks per program
STATS = {"triton": 0, "bnb": 0}
_CODE = {}
_BROKEN = False


@triton.jit
def _nf4_tile(A, Q, CODE2, S2, OFF, CODE, OUT, n_bytes, n_blk, n_out,
              BS2: tl.constexpr, NBLK: tl.constexpr, BPS2: tl.constexpr,
              NESTED: tl.constexpr):
    pid = tl.program_id(0)
    rb = pid * NBLK + tl.arange(0, NBLK)                  # quantisation block ids
    bm = rb < n_blk
    if NESTED:
        q = tl.load(Q + rb, mask=bm, other=0).to(tl.int32)
        s2 = tl.load(S2 + rb // BPS2, mask=bm, other=0.0)
        scale = tl.load(CODE2 + q, mask=bm, other=0.0) * s2 + tl.load(OFF)
    else:
        scale = tl.load(Q + rb, mask=bm, other=0.0)
    c = tl.arange(0, BS2)
    j = tl.arange(0, 2)
    byte = rb[:, None, None] * BS2 + c[None, :, None]     # [NBLK, BS2, 1]
    b = tl.load(A + byte, mask=byte < n_bytes, other=0).to(tl.int32)
    nib = (b >> ((1 - j[None, None, :]) * 4)) & 15         # [NBLK, BS2, 2], high nibble first
    val = tl.load(CODE + nib) * scale[:, None, None]
    o = byte * 2 + j[None, None, :]                       # output element index
    tl.store(OUT + o, val.to(OUT.dtype.element_ty), mask=o < n_out)


def _code(device):
    key = str(device)
    if key not in _CODE:
        import bitsandbytes.functional as BF
        _CODE[key] = BF.get_4bit_type("nf4", device=device).to(torch.float32).contiguous()
    return _CODE[key]


def _aux(st, device):
    a = getattr(st, "_soup_nf4", None)
    if a is not None:
        return a
    if st.nested:
        s2 = st.state2
        a = dict(
            Q=st.absmax.contiguous(),
            S2=s2.absmax.float().contiguous(),
            CODE2=s2.code.float().contiguous(),
            OFF=torch.as_tensor(st.offset, dtype=torch.float32, device=device).reshape(1).contiguous(),
            BPS2=int(s2.blocksize), NESTED=True)
    else:
        q = st.absmax.float().contiguous()
        a = dict(Q=q, S2=q, CODE2=q, OFF=q, BPS2=1, NESTED=False)
    try:
        st._soup_nf4 = a  # cache: the weights are frozen
    except Exception:  # noqa: BLE001
        pass
    return a


def dequantize_nf4(A, st):
    bs = int(st.blocksize)
    bs2 = bs // 2
    if bs2 & (bs2 - 1):
        raise ValueError("blocksize must be a power of two")
    numel = st.shape.numel()
    flat = A.reshape(-1)
    out = torch.empty(st.shape, dtype=st.dtype, device=A.device)
    n_blk = triton.cdiv(numel, bs)
    a = _aux(st, A.device)
    _nf4_tile[(triton.cdiv(n_blk, NBLK),)](
        flat, a["Q"], a["CODE2"], a["S2"], a["OFF"], _code(A.device), out,
        flat.numel(), n_blk, numel,
        BS2=bs2, NBLK=NBLK, BPS2=a["BPS2"], NESTED=a["NESTED"], num_warps=4)
    return out


def _report():
    print(f"[solution] nf4 dequant calls: triton={STATS['triton']} "
          f"bnb={STATS['bnb']} broken={_BROKEN}")


def patch_bnb_dequant():
    import bitsandbytes.functional as BF

    orig = BF.dequantize_4bit
    if getattr(orig, "_soup_fused", False):
        return

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=64,
                        quant_type="fp4"):
        global _BROKEN
        st = quant_state
        if (not _BROKEN and st is not None and absmax is None and out is None
                and A.is_cuda and A.dtype == torch.uint8
                and getattr(st, "quant_type", None) == "nf4"
                and st.dtype in (torch.float16, torch.float32)
                and st.blocksize % 2 == 0 and st.shape.numel() == 2 * A.numel()
                and 2 * A.numel() < 2 ** 31):
            try:
                res = dequantize_nf4(A, st)
                STATS["triton"] += 1
                return res.t() if A.shape[0] == 1 else res
            except Exception:  # never break training because of the speed-up
                _BROKEN = True
                print("[solution] nf4: Triton path failed, bitsandbytes from now on:")
                traceback.print_exc()
        STATS["bnb"] += 1
        return orig(A, quant_state, absmax, out, blocksize, quant_type)

    dequantize_4bit._soup_fused = True
    BF.dequantize_4bit = dequantize_4bit
    for name, mod in list(sys.modules.items()):
        if name.startswith("bitsandbytes") and getattr(mod, "dequantize_4bit", None) is orig:
            mod.dequantize_4bit = dequantize_4bit
    atexit.register(_report)
