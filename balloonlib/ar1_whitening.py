"""AR(1) whitening of the BOLD data term (Cochrane-Orcutt / Prais-Winsten).

What it does, in four steps
---------------------------
1. residual  r = bold_pinn_sampled - Bold_data,  offset = mean(r),  r = r - offset
   (same as the current loss, which adds -mean(r): only the sign convention differs)
2. every `every` iterations after `warmup`: rho = lag-1 autocorrelation of r.
   If rho is not significantly > 0 (one-sided 5 % test, rho > 1.645/sqrt(n)), use rho = 0.
3. whiten:  u_1 = r_1,   u_k = (r_k - rho r_{k-1}) / sqrt(1 - rho^2)
   (each u_k is the part of r_k NOT predictable from r_{k-1}, rescaled to the noise variance)
4. bold_loss = meFn(u, 0)          (= mean(u^2) for nn.MSELoss; rho = 0 gives the current loss)

Why it limits fitting to noise: AR(1) noise with rho > 0 is slow.  Whitening down-weights slow
residual fluctuations - the ones the noise model already explains - so the network is not
rewarded for bending the HRF to follow them.

References: Cochrane & Orcutt 1949 (J Am Stat Assoc 44:32); Prais & Winsten 1954;
Bullmore et al. 1996 (Magn Reson Med 35:261); Woolrich et al. 2001 (NeuroImage 14:1370).
"""
from __future__ import annotations

import math
import torch

__all__ = ["AR1Whitening"]


class AR1Whitening:
    """Put in ``data_params["noise_model"]``; ``training.loss`` / ``train`` call it.

    TR : repetition time (s); used only for the bound rho_max = exp(-TR / tau_max).
    warmup : first iteration at which rho is estimated (fit must be near converged:
        model misfit is slow and would be read as autocorrelated noise).
    every : re-estimation period.  freeze_after : stop re-estimating after this iteration.
    tau_max : longest noise correlation time allowed (s).
    """

    def __init__(self, TR: float, warmup: int = 1000, every: int = 250,
                 freeze_after: int | None = None, tau_max: float = 5.0):
        self.TR, self.warmup, self.every, self.freeze_after = float(TR), warmup, every, freeze_after
        self.rho_max = math.exp(-TR / tau_max)
        self.rho = 0.0
        self.history: list[dict] = []
        self._r = None

    def whiten(self, r: torch.Tensor) -> torch.Tensor:
        if self.rho == 0.0:
            return r
        return torch.cat([r[:1], (r[1:] - self.rho * r[:-1]) / math.sqrt(1.0 - self.rho ** 2)])

    def loss(self, bold_pred, bold_data, t, meFn):
        r = bold_pred.reshape(-1) - bold_data.reshape(-1).to(bold_pred.dtype)
        offset = r.mean().detach()                      # offset = mean(r)
        r = r - offset                                  # mean-centred residual
        self._r = r.detach()
        u = self.whiten(r)
        return meFn(u, torch.zeros_like(u))

    def due(self, i: int) -> bool:
        if i < self.warmup or (i - self.warmup) % self.every:
            return False
        return self.freeze_after is None or i <= self.freeze_after

    @torch.no_grad()
    def update(self, i: int) -> dict | None:
        if self._r is None or not self.due(i):
            return None
        r = self._r.double()
        n = r.numel()
        r1 = float((r[1:] @ r[:-1]) / (r @ r))           # lag-1 autocorrelation
        white = r1 <= 1.645 / math.sqrt(n)               # H0: white noise, one-sided 5 %
        self.rho = 0.0 if white else min(r1, self.rho_max)
        u = self.whiten(r)
        ud = u - u.mean()
        est = {"iter": int(i), "lag1": r1, "phi": self.rho, "phi_raw": r1, "white": bool(white),
               "tau": (-self.TR / math.log(self.rho)) if self.rho > 0 else 0.0,
               "sigma": float(u.std()), "lr": float("nan"),
               "lag1_whitened": float((ud[1:] @ ud[:-1]) / (ud @ ud)),
               "at_bound": r1 > self.rho_max}
        self.history.append(est)
        return est

    def state(self) -> dict:
        return {"method": "AR(1) lag-1 (Cochrane-Orcutt)", "phi": self.rho, "TR": self.TR,
                "rho_max": self.rho_max, "history": list(self.history)}
