"""GPTQ fake-quantization, hand-rolled.

No auto-gptq / gptqmodel dependency: quantization happens in-memory on the
live model, exactly like 02's RTN pass, so the result is evaluated with the
live-model path in measure.run_lm_eval (the same trick AWQ forced on us).

Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers" (2022). Structure follows the original
IST-DASLab implementation, simplified: no act_order, no packing, fp32 math.

Drop at src/effml/gptq.py
"""

import math

import torch


# --------------------------------------------------------------------------
# shared quant grid  (identical math to 02's rtn_quantize_, factored out)
# --------------------------------------------------------------------------

def find_qparams(W: torch.Tensor, n_bits: int = 4):
    """Asymmetric min/max grid over the last dim.

    W: (..., g) fp32. Returns (scale, zero), each (..., 1), broadcastable.
    """
    w_max = W.amax(dim=-1, keepdim=True)
    w_min = W.amin(dim=-1, keepdim=True)
    qmax = 2 ** n_bits - 1
    scale = (w_max - w_min).clamp(min=1e-8) / qmax
    zero = (-w_min / scale).round()
    return scale, zero


def quant_dequant(W: torch.Tensor, scale, zero, n_bits: int) -> torch.Tensor:
    qmax = 2 ** n_bits - 1
    Wq = (W / scale + zero).round().clamp(0, qmax)
    return (Wq - zero) * scale


# --------------------------------------------------------------------------
# per-Linear GPTQ state
# --------------------------------------------------------------------------

class GPTQ:
    """Accumulates the Hessian for one Linear, then quantizes it in place."""

    def __init__(self, layer: torch.nn.Linear, n_bits: int = 4, group_size: int = 128):
        self.layer = layer
        self.n_bits = n_bits
        self.group_size = group_size

        W = layer.weight.data
        self.rows, self.columns = W.shape
        self.dev = W.device
        # H is (in_features, in_features) fp32 -- the big memory item.
        # Qwen3-4B down_proj: 9728^2 * 4B = 378 MB.
        self.H = torch.zeros((self.columns, self.columns),
                             device=self.dev, dtype=torch.float32)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """inp: (..., in_features) activations entering this Linear."""
        inp = inp.reshape(-1, inp.shape[-1]).t().float()   # (cols, n_tokens)
        n = inp.shape[1]
        # running mean of 2 * X X^T
        self.H *= self.nsamples / (self.nsamples + n)
        self.nsamples += n
        inp = inp * math.sqrt(2.0 / self.nsamples)
        self.H += inp.matmul(inp.t())

    @torch.no_grad()
    def quantize(self, percdamp: float = 0.01, blocksize: int = 128) -> float:
        """Quantize the weight in place. Returns the proxy loss (lower = better)."""
        W = self.layer.weight.data.clone().float()
        H = self.H

        # dead input channels: nothing ever activated them, so their weights
        # are unconstrained -- zero them and give H a unit diagonal entry.
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # dampening: keeps the Cholesky well-conditioned
        damp = percdamp * torch.mean(torch.diag(H))
        idx = torch.arange(self.columns, device=self.dev)
        H[idx, idx] += damp

        # Hinv as an upper-triangular Cholesky factor of H^-1
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        Q = torch.zeros_like(W)
        total_loss = 0.0
        scale = zero = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                col = i1 + i
                w = W1[:, i]
                d = Hinv1[i, i]

                # refresh the group grid every group_size columns.
                # NOTE: qparams come from W (block-level error compensation
                # applied) not W1 (within-block). Matches upstream; exact when
                # group_size is a multiple of blocksize, which it is for
                # g128/b128 and g64,g32 nested inside b128.
                if col % self.group_size == 0:
                    g_end = min(col + self.group_size, self.columns)
                    scale, zero = find_qparams(W[:, col:g_end], self.n_bits)

                q = quant_dequant(w.unsqueeze(1), scale, zero, self.n_bits).flatten()
                Q1[:, i] = q

                total_loss += ((w - q) ** 2 / d ** 2).sum().item() / 2
                err = (w - q) / d
                # push this column's error onto the columns still to come
                W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        return total_loss

    def free(self):
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# block-sequential driver
# --------------------------------------------------------------------------

@torch.no_grad()
def gptq_quantize_model(model, calib_ids, n_bits=4, group_size=128,
                        percdamp=0.01, blocksize=128, verbose=True):
    """Quantize every decoder-block Linear in place, block by block.

    model:     a loaded causal LM, ALL ON ONE DEVICE (see note in 04).
    calib_ids: list of (1, seqlen) LongTensors of calibration token ids.

    lm_head is never touched -- Qwen3 ties it to the input embeddings.
    """
    dev = next(model.parameters()).device
    prev_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    # ---- capture the inputs to block 0 -----------------------------------
    inps, fwd_kwargs = [], {}

    class Catcher(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, hidden_states, **kwargs):
            inps.append(hidden_states)
            # position_embeddings (rotary cos/sin), attention_mask, etc.
            # These are identical across equal-length batches, so one snapshot
            # is enough. Cache objects are stateful -- drop them.
            for k, v in kwargs.items():
                if "past_key_value" in k or k == "cache_position":
                    continue
                fwd_kwargs[k] = v
            raise _StopForward

    class _StopForward(Exception):
        pass

    layers[0] = Catcher(layers[0])
    for ids in calib_ids:
        try:
            model(ids.to(dev))
        except _StopForward:
            pass
    layers[0] = layers[0].mod

    if not inps:
        raise RuntimeError("Catcher captured nothing -- check model.model.layers path")
    if verbose:
        print(f"[gptq] captured {len(inps)} calib samples, "
              f"kwargs={sorted(fwd_kwargs)}")

    outs = [torch.zeros_like(x) for x in inps]

    # ---- walk the blocks -------------------------------------------------
    for bi, layer in enumerate(layers):
        subset = {n: m for n, m in layer.named_modules()
                  if isinstance(m, torch.nn.Linear)}

        gptq = {n: GPTQ(m, n_bits, group_size) for n, m in subset.items()}

        def make_hook(name):
            def hook(_mod, inp, _out):
                gptq[name].add_batch(inp[0].data)
            return hook

        handles = [m.register_forward_hook(make_hook(n)) for n, m in subset.items()]
        for j, x in enumerate(inps):
            layer(x, **fwd_kwargs)
        for h in handles:
            h.remove()

        losses = {}
        for n in subset:
            losses[n] = gptq[n].quantize(percdamp=percdamp, blocksize=blocksize)
            gptq[n].free()

        # re-run with the QUANTIZED weights so the next block calibrates on
        # the activations it will actually see. This is where GPTQ's edge
        # over RTN mostly comes from.
        for j, x in enumerate(inps):
            out = layer(x, **fwd_kwargs)
            outs[j] = out[0] if isinstance(out, tuple) else out

        inps, outs = outs, inps

        if verbose:
            worst = max(losses, key=losses.get)
            print(f"[gptq] block {bi:>2}/{len(layers) - 1}  "
                  f"{len(subset)} linears  worst={worst} ({losses[worst]:.1f})")

    model.config.use_cache = prev_cache
    return model