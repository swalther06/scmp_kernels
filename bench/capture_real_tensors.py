"""Capture real (activation, weight) operand pairs from a HF model's Linear
layers, for kernel-level SC MSE profiling.

For each selected decoder layer / projection we hook the nn.Linear and save
``(a, b)`` where ``a`` is the flattened layer input ``[N, D]`` and ``b`` is the
weight ``[M, D]`` — so ``sc_matmul(a, b) == F.linear(x, W)`` (i.e. ``a @ b.T``).

Output: a torch .pt holding ``{ "<layer>.<proj>": (a_fp16, b_fp16) }``.

    python bench/capture_real_tensors.py --layers 0,15,31 --out captured.pt
"""
from __future__ import annotations
import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("CAP_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    ap.add_argument("--prompt", default="Explain what a large language model is, in detail.")
    ap.add_argument("--layers", default="0,15,31", help="comma-sep decoder layer indices")
    ap.add_argument("--out", default="captured.pt")
    args = ap.parse_args()

    want = {int(x) for x in args.layers.split(",")}
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    captured: dict[str, tuple] = {}
    hooks = []

    def mk(name):
        def hook(mod, inp, _out):
            if name in captured:
                return
            x = inp[0].detach()
            a = x.reshape(-1, x.shape[-1]).to(torch.float16).cpu().contiguous()
            b = mod.weight.detach().to(torch.float16).cpu().contiguous()
            captured[name] = (a, b)
        return hook

    layers = model.model.layers
    for li in want:
        for pn in PROJ:
            for mname, m in layers[li].named_modules():
                if mname.endswith(pn) and isinstance(m, torch.nn.Linear):
                    hooks.append(m.register_forward_hook(mk(f"L{li}.{pn}")))
                    break

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()

    for k, (a, b) in sorted(captured.items()):
        print(f"  {k:18s} a={tuple(a.shape)} b={tuple(b.shape)}")
    torch.save(captured, args.out)
    print(f"saved {len(captured)} operand pairs -> {args.out}")


if __name__ == "__main__":
    main()
