"""Check that our AGCRN port (code/models/agcrn.py) gives the SAME forward output and gradients
as the official implementation (https://github.com/LeiBAI/AGCRN).

Run with the BasicTS Python 3.11 environment (needs torch):
  /content/venv/bin/python code/scripts/check_agcrn_equivalence.py --official /content/AGCRN_official
Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "BasicTS" / "src"))
sys.path.insert(0, str(ROOT / "code"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True, help="path of the cloned LeiBAI/AGCRN repository")
    a = ap.parse_args()
    sys.path.insert(0, a.official)

    import torch
    from torch import nn
    from model.AGCRN import AGCRN as OfficialAGCRN          # official code (package 'model')
    from models.agcrn import AGCRN, AGCRNConfig             # our port (package 'models')

    ok = True
    for cheb_k in (2, 3):
        torch.manual_seed(0)
        n, t_in, horizon, embed, units, layers = 20, 12, 12, 6, 16, 2
        args = types.SimpleNamespace(num_nodes=n, input_dim=1, rnn_units=units, output_dim=1, horizon=horizon,
                                     num_layers=layers, default_graph=True, embed_dim=embed, cheb_k=cheb_k)
        official = OfficialAGCRN(args)
        for p in official.parameters():                      # official Run.py initialisation
            nn.init.xavier_uniform_(p) if p.dim() > 1 else nn.init.uniform_(p)

        ours = AGCRN(AGCRNConfig(input_len=t_in, output_len=horizon, num_features=n, embed_dim=embed,
                                 rnn_units=units, num_layers=layers, cheb_k=cheb_k))
        ours.load_state_dict(official.state_dict(), strict=True)   # identical parameter names/shapes
        official.eval()
        ours.eval()

        x = torch.randn(4, t_in, n)
        out_off = official(x.unsqueeze(-1), None).squeeze(-1)      # official input: [B, T, N, 1]
        out_our = ours(x)
        fwd = (out_off - out_our).abs().max().item()

        official.zero_grad()
        ours.zero_grad()
        out_off.abs().mean().backward()
        out_our.abs().mean().backward()
        worst = 0.0
        g_off = dict(official.named_parameters())
        for name, p in ours.named_parameters():
            d = (p.grad - g_off[name].grad).abs().max().item()
            scale = g_off[name].grad.abs().max().item() + 1e-12
            worst = max(worst, d / scale)

        passed = fwd < 1e-5 and worst < 1e-4
        ok &= passed
        print(f"cheb_k={cheb_k}: max|forward diff|={fwd:.2e}  worst relative grad diff={worst:.2e}  "
              f"output shape={tuple(out_our.shape)}  -> {'PASS' if passed else 'FAIL'}")
    print("RESULT:", "PASS (port is equivalent to the official code)" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
