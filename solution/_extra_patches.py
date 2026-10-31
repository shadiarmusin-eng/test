"""Glue: call apply_extra() at the END of solution.apply() (after the example's patch)."""
import gc
import warnings

_DONE = False
TUNE_GC = True  # fewer Python garbage-collector pauses on the CPU side; no effect on values


def apply_extra():
    global _DONE
    if _DONE:
        return
    try:
        import torch
        if not torch.cuda.is_available():  # Soup's unit tests may run without a GPU
            return
        from . import fused_ce, head_loss, lora_epilogue, nf4, rmsnorm, unpad
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"solution extras not loaded: {e}")
        return
    for fn in (rmsnorm.patch_qwen2_rmsnorm, rmsnorm.patch_fp16_norm_outputs,
               rmsnorm.patch_decoder_add_norm, rmsnorm.patch_peft_input_cast,
               fused_ce.patch_loss, fused_ce.patch_entropy,
               nf4.patch_bnb_dequant, head_loss.patch_head_loss,
               lora_epilogue.patch_lora_epilogue,
               unpad.patch_mlp_unpad):
        try:
            fn()
            print(f"[solution] OK: {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"[solution] SKIPPED {fn.__name__}: {e}")
    if TUNE_GC:
        gc.collect()
        gc.freeze()
        gc.set_threshold(100_000, 50, 100)
    _DONE = True
