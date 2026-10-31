import os, time

def install():
    if os.environ.get("SOUP_PROF") != "1":
        return
    import torch
    from transformers import Trainer
    from torch.profiler import profile, ProfilerActivity

    orig = Trainer.training_step
    st = {"n": 0, "prof": None, "t0": None, "pad": []}

    def training_step(self, model, inputs, *a, **k):
        st["n"] += 1
        n = st["n"]
        try:
            st["pad"].append(1.0 - inputs["attention_mask"].float().mean().item())
        except Exception:
            pass
        if n == 4:
            torch.cuda.synchronize()
            st["t0"] = time.time()
            st["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            st["prof"].__enter__()
        out = orig(self, model, inputs, *a, **k)
        if n == 8 and st["prof"] is not None:
            torch.cuda.synchronize()
            wall = time.time() - st["t0"]
            prof, st["prof"] = st["prof"], None
            prof.__exit__(None, None, None)
            ka = prof.key_averages()
            try:
                tab = ka.table(sort_by="self_cuda_time_total", row_limit=30, max_name_column_width=60)
            except Exception:
                tab = ka.table(sort_by="self_device_time_total", row_limit=30, max_name_column_width=60)
            with open("/kaggle/working/prof.txt", "w") as f:
                f.write(f"wall, steps 4-8 (5 steps): {wall:.2f}s\n")
                f.write(f"mean pad fraction: {sum(st['pad'])/max(1,len(st['pad'])):.3f}\n")
                f.write(tab)
        return out

    Trainer.training_step = training_step
