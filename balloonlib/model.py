"""
Neural network model for the BalloonLib PINN framework.

Provides the :class:`Multihead` physics-informed neural network that
simultaneously predicts the four Balloon haemodynamic state variables.
"""

import torch
import torch.nn as nn

from balloonlib.layers import FourierFeatureMapping, FactorizedLinear
from balloonlib.physics import dfdt

# Global dtype (mirrors balloonpinnlib globals so Multihead methods are consistent)
dtype = torch.float32


class Multihead(nn.Module):
    """Multi-head Physics-Informed Neural Network for the Balloon haemodynamic model.

    The network simultaneously predicts the four state variables of the Balloon model:
    normalised cerebral blood flow (*f*), cerebral metabolic rate of oxygen (*m*),
    blood volume (*v*), and deoxyhaemoglobin content (*q*).
    It optionally uses Fourier feature encoding and Random Weight Factorization (RWF)
    layers for improved training of high-frequency physiological signals.
    """

    def __init__(
        self,
        Core2NV: bool = False,
        mode: str = "grad",
        impulse: bool = False,
        act=nn.Softmax(dim=1),
        nv_fn: tuple = (nn.Softplus(), nn.Softplus()),
        core_fn: tuple = (nn.Softplus(), nn.Softplus()),
        dtype=torch.float32,
        ic_clamp: dict = {},
        use_fourier: bool = False,
        fourier_mapping_size: int = 9,
        fourier_scale: float = 0.85,
        fourier_learnable: bool = True,
        multi_scale_fourier: bool = False,
        random_weightsMatrix: bool = True,
        seed=None,
    ):
        """Initialise the Multihead PINN.

        Parameters
        ----------
        Core2NV : bool
            If ``True``, feeds core-layer hidden state back into the
            neurovascular (NV) sub-network.
        mode : str
            ``'grad'`` propagates gradients through the shared encoder;
            ``'detach'`` runs the encoder without gradients.
        impulse : bool
            If ``True``, the stimulus impulse is appended to the time input
            (network input dimension becomes 2).
        act : nn.Module
            Activation function applied after each shared hidden layer.
        nv_fn : tuple[nn.Module, nn.Module]
            Pair ``(act_f, act_m)`` applied to the NV output heads *f* and *m*.
            Defaults to ``(Softplus, Softplus)``.
        core_fn : tuple[nn.Module, nn.Module]
            Pair ``(act_v, act_q)`` applied to the core output heads *v* and *q*.
        dtype : torch.dtype
            Floating-point precision for all parameters.
        ic_clamp : dict
            Optional dict with keys ``'set'`` (int, number of IC samples) and
            ``'band'`` (float, half-width for soft-clamp around 1.0).
        use_fourier : bool
            If ``True``, applies Fourier feature encoding to the time input.
        fourier_mapping_size : int
            Number of random Fourier features.
        fourier_scale : float
            Standard deviation of the Gaussian used to sample the frequency matrix.
        fourier_learnable : bool
            If ``True``, Fourier frequencies are trainable parameters.
        multi_scale_fourier : bool
            If ``True``, concatenates three Fourier encoders at scales
            ``0.5x``, ``1x``, and ``2x`` ``fourier_scale``.
        random_weightsMatrix : bool
            If ``True``, uses :class:`~balloonlib.layers.FactorizedLinear` (RWF)
            instead of :class:`torch.nn.Linear` for the hidden layers.
        seed : int or None
            Random seed for reproducibility.  ``None`` disables seeding.
        """
        super().__init__()
        self.act = act

        if nv_fn is not None:
            self.nv_fn = nv_fn
        else:
            self.nv_fn = (nn.Softplus(), nn.Softplus())

        self.core_fn = core_fn
        self.Core2NV = Core2NV
        self.impulse = impulse
        self.mode = mode
        self.use_fourier = use_fourier
        self.multi_scale_fourier = multi_scale_fourier
        self.ic_clamp = ic_clamp
        self.random_weightsMatrix = random_weightsMatrix

        if seed is not None:
            self.seed = seed
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)

        # Balloon model constants
        self.DEFAULT_PARAMS = {
            "E_0": torch.tensor(
                0.32, dtype=dtype
            ),  # nn.Parameter(torch.tensor(0.32, dtype=dtype)),
            "V_0": torch.tensor(
                0.03, dtype=dtype
            ),  # nn.Parameter(torch.tensor(0.03, dtype=dtype)),
            "TE": torch.tensor(0.06, dtype=dtype),  # nn.Parameter(torch.tensor(0.06, dtype=dtype)),
            "O_0": torch.tensor(
                40.3, dtype=dtype
            ),  # nn.Parameter(torch.tensor(40.3, dtype=dtype)),
            "r_0": torch.tensor(25, dtype=dtype),  # nn.Parameter(torch.tensor(25, dtype=dtype)),
            "epsilon": torch.tensor(
                1.43, dtype=dtype
            ),  # nn.Parameter(torch.tensor(1.43, dtype=dtype)),
        }

        # Grubb's exponent (fixed)
        self.alpha = nn.Parameter(torch.tensor(0.4, dtype=dtype))  # , 0.4 #

        # Davis (1998) BOLD model parameters
        self.Davis_params = {
            "alpha": self.alpha,
            "beta": nn.Parameter(torch.tensor(1.5, dtype=dtype)),
            "A": nn.Parameter(torch.tensor(0.075, dtype=dtype)),
        }

        # Determine input dimension
        base_input_dim = 2 if self.impulse else 1

        if self.use_fourier:
            if self.multi_scale_fourier:
                self.fourier_low = FourierFeatureMapping(
                    base_input_dim,
                    fourier_mapping_size // 3,
                    scale=fourier_scale * 0.5,
                    learnable=fourier_learnable,
                )
                self.fourier_mid = FourierFeatureMapping(
                    base_input_dim,
                    fourier_mapping_size // 3,
                    scale=fourier_scale,
                    learnable=fourier_learnable,
                )
                self.fourier_high = FourierFeatureMapping(
                    base_input_dim,
                    fourier_mapping_size // 3,
                    scale=fourier_scale * 2.0,
                    learnable=fourier_learnable,
                )
                first_layer_input = 3 * 2 * (fourier_mapping_size // 3)
            else:
                self.fourier_mapping = FourierFeatureMapping(
                    base_input_dim,
                    fourier_mapping_size,
                    scale=fourier_scale,
                    learnable=fourier_learnable,
                )
                first_layer_input = 2 * fourier_mapping_size
        else:
            first_layer_input = base_input_dim

        if not self.random_weightsMatrix:
            self.linear = {
                "linear1": nn.Linear(first_layer_input, 128, bias=False, dtype=dtype),
                # "linear2": nn.Linear(128, 128, bias=False, dtype=dtype),
                "linear3": nn.Linear(128, 256, bias=False, dtype=dtype),
                # "linear4": nn.Linear(256, 256, bias=False, dtype=dtype),
                "linear5": nn.Linear(256, 512, bias=False, dtype=dtype),
            }
            self.nv_final_layers = nn.ModuleList(
                [nn.Linear(256, 1, bias=True, dtype=dtype) for _ in range(2)]
            )
            self.Core = nn.ModuleList(
                [nn.Linear(256 + (1 + i), 1, bias=True, dtype=dtype) for i in range(2)]
            )
        else:
            self.linear = {
                "linear1": FactorizedLinear(first_layer_input, 128, bias=False, dtype=dtype),
                "linear3": FactorizedLinear(128, 256, bias=False, dtype=dtype),
                "linear5": FactorizedLinear(256, 512, bias=False, dtype=dtype),
            }
            self.nv_final_layers = nn.ModuleList(
                # [FactorizedLinear(256, 1, bias=True, dtype=dtype) for _ in range(2)]
                [nn.Linear(256, 1, bias=True, dtype=dtype) for _ in range(2)]
            )
            self.Core = nn.ModuleList(
                # [FactorizedLinear(257 + i, 1, bias=True, dtype=dtype) for i in range(2)]
                [nn.Linear(256 + (1 + i), 1, bias=True, dtype=dtype) for i in range(2)]
            )

        self.Sequential = nn.Sequential(
            *[
                nn.Sequential(linear, self.act)
                for linear in nn.ModuleList(list(self.linear.values()))
            ]
        )
        # neural network splitting? False = [0,0]; True = [0,1]
        # nns = [0,0] means f & m from outi[0] v & q from outi[1]
        self.nns = [0, 1 if self.Core2NV else 0]

    @property
    def epsilon(self):
        """Clamped intra/extravascular BOLD ratio (non-negative)."""
        return torch.clamp(self.DEFAULT_PARAMS["epsilon"], min=0)

    def init_nn_params(self):
        """Initialise network weights using Xavier/Glorot uniform initialisation.

        Only :class:`torch.nn.Linear` layers are initialised; :class:`FactorizedLinear`
        layers perform their own RWF initialisation at construction time.
        Output head biases are set to ``softplus⁻¹(1) ≈ 0.541`` so that the
        Softplus-activated outputs start at the physiological baseline of 1.0,
        matching the initial condition target for f, m, v, and q.
        """

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

        # softplus(x) = 1.0 when x = log(exp(1) - 1) ≈ 0.541
        softplus_inv_1 = 0.5413722

        for layer in list(self.nv_final_layers) + list(self.Core):
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.constant_(layer.bias, softplus_inv_1)

    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature encoding to the network input.

        Parameters
        ----------
        x : torch.Tensor
            Raw input of shape ``(N, 1)`` (time) or ``(N, 2)`` (time + stimulus).

        Returns
        -------
        torch.Tensor
            Encoded features ready for the shared backbone.
        """
        if not self.use_fourier:
            return x

        if self.multi_scale_fourier:
            f_low = self.fourier_low(x)
            f_mid = self.fourier_mid(x)
            f_high = self.fourier_high(x)
            return torch.cat([f_low, f_mid, f_high], dim=-1)
        else:
            return self.fourier_mapping(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the multi-head network.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(N, 1)`` for time-only or ``(N, 2)`` for
            time + stimulus.

        Returns
        -------
        output : torch.Tensor
            Concatenated state variables ``[f, m, v, q]``, shape ``(N, 4)``.
        out : torch.Tensor
            Final hidden representation from the shared backbone, shape ``(N, 512)``.
        """
        x_encoded = self.encode_input(x)

        if self.mode == "detach":
            with torch.no_grad():
                out = self.Sequential(x_encoded)
        else:
            out = self.Sequential(x_encoded)

        out_half = out.shape[1] // 2
        outi = [out[:, :out_half], out[:, out_half:]]

        # nns = [0,0] means f & m from outi[0]
        # nns = [0,1] means f from outi[0] & m from outi[1]
        self.f, self.m = [
            self.nv_fn[i](self.nv_final_layers[i](outi[self.nns[i]])) for i in range(2)
        ]

        if self.ic_clamp:

            def soft_clamp(x, center=1.0, half_width=self.ic_clamp["band"]):
                """Differentiable symmetric clamp around *center* ± *half_width*."""
                return center + half_width * torch.tanh((x - center) / half_width)

            mask = torch.zeros(self.f.shape[0], dtype=torch.bool, device=self.f.device)
            mask[: self.ic_clamp["set"]] = True
            self.f = torch.where(mask.unsqueeze(1), soft_clamp(self.f), self.f)
            self.m = torch.where(mask.unsqueeze(1), soft_clamp(self.m), self.m)

        # if self.Core2NV True:
        #   nns = [0,1] -> v_arg = [outi[0] ,f]
        # else: v_arg = [outi[1] ,f]
        v_arg = torch.cat((outi[1 - self.nns[1]], self.f), axis=1)
        q_arg = torch.cat((outi[1], self.m, self.core_fn[0](self.Core[0](v_arg))), axis=1)
        core_arg = [v_arg, q_arg]

        self.v, self.q = [self.core_fn[i](self.Core[i](core_arg[i])) for i in range(2)]

        output = torch.cat((self.f, self.m, self.v, self.q), axis=1)
        return output, out

    # @torch.compile
    def fout(
        self,
        v: torch.Tensor = None,
        alpha=None,
        tau_m: float = 20,
        dvdt: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute the balloon outflow function.

        Parameters
        ----------
        v : torch.Tensor, optional
            Normalised blood volume.  Defaults to ``self.v``.
        alpha : float
            Grubb's stiffness exponent.
        tau_m : float
            Viscoelastic time constant (s).
        dvdt : torch.Tensor, optional
            Time derivative of *v*.  Defaults to zero (purely elastic regime).

        Returns
        -------
        torch.Tensor
            Outflow tensor of shape ``(N, 1)``.
        """
        if v is None:
            v = self.v.squeeze() if self.v.ndim >= 2 else self.v
        else:
            v = v.squeeze() if v.ndim >= 2 else v

        if dvdt is None:
            dvdt = torch.zeros_like(v)
        else:
            dvdt = dvdt.squeeze() if dvdt.ndim >= 2 else dvdt

        if alpha is None:
            alpha = self.alpha

        v_safe = v.clamp(min=1e-8)
        return (torch.exp(torch.log(v_safe) / alpha) + (tau_m * dvdt)).view(-1, 1)

    def predictor(
        self,
        predict_v: torch.Tensor = None,
        predict_q: torch.Tensor = None,
        params: dict = None,
        linear: bool = False,
    ) -> torch.Tensor:
        """Predict the BOLD signal from current haemodynamic state.

        Implements the Buxton (2004) / Stephan (2007) BOLD signal equation.

        Parameters
        ----------
        predict_v : torch.Tensor, optional
            Normalised blood volume *v*.  Defaults to ``self.v``.
        predict_q : torch.Tensor, optional
            Normalised deoxyhaemoglobin content *q*.  Defaults to ``self.q``.
        params : dict, optional
            Override any key in :attr:`DEFAULT_PARAMS`.
        linear : bool
            If ``True``, uses the linearised balloon approximation.

        Returns
        -------
        torch.Tensor
            Predicted fractional BOLD signal change ΔS/S₀.
        """
        if predict_v is None:
            predict_v = self.v
        if predict_q is None:
            predict_q = self.q

        p = {**self.DEFAULT_PARAMS, **(params or {})}

        self.k_1 = (1 - p["V_0"]) * 4.3 * p["O_0"] * p["E_0"] * p["TE"]
        self.k_2 = 2.0 * p["E_0"]
        self.k_3 = 1.0 - self.epsilon

        if not linear:
            return p["V_0"] * (
                self.k_1 * (1.0 - predict_q)
                + self.k_2 * (1.0 - predict_q / predict_v)
                + self.k_3 * (1.0 - predict_v)
            )
        else:
            return p["V_0"] * (
                (self.k_1 + self.k_2) * (1.0 - predict_q)
                + (self.k_3 - self.k_2) * (1.0 - predict_v)
            )

    def dpredt(
        self,
        v: torch.Tensor = None,
        q: torch.Tensor = None,
        dv: torch.Tensor = None,
        dq: torch.Tensor = None,
        t: torch.Tensor = None,
        params: dict = None,
        linear: bool = False,
    ) -> torch.Tensor:
        """Compute the time derivative of the BOLD signal predictor.

        Parameters
        ----------
        v : torch.Tensor, optional
            Normalised blood volume *v*.  Defaults to ``self.v``.
        q : torch.Tensor, optional
            Normalised deoxyhaemoglobin *q*.  Defaults to ``self.q``.
        dv : torch.Tensor, optional
            Time derivative of *v*.  Computed via autograd if ``None``.
        dq : torch.Tensor, optional
            Time derivative of *q*.  Computed via autograd if ``None``.
        t : torch.Tensor, optional
            Time tensor used for autograd when ``dv`` or ``dq`` is ``None``.
        params : dict, optional
            Override any key in :attr:`DEFAULT_PARAMS`.
        linear : bool
            If ``True``, evaluates the linearised balloon approximation.

        Returns
        -------
        torch.Tensor
            Time derivative d(BOLD)/dt with the same shape as ``dv``.
        """
        p = {**self.DEFAULT_PARAMS, **(params or {})}

        self.k_1 = (1 - p["V_0"]) * 4.3 * p["O_0"] * p["E_0"] * p["TE"]
        self.k_2 = 2.0 * p["E_0"]
        self.k_3 = 1.0 - self.epsilon

        if v is None:
            v = self.v.squeeze() if self.v.ndim >= 2 else self.v
        if q is None:
            q = self.q.squeeze() if self.q.ndim >= 2 else self.q

        if dv is None:
            dv = dfdt(signal=v, arg=t)
        if dq is None:
            dq = dfdt(signal=q, arg=t)

        shape = dv.size()
        dv = dv.squeeze() if dv.ndim >= 2 else dv
        dq = dq.squeeze() if dq.ndim >= 2 else dq

        if not linear:
            return (
                -p["V_0"]
                * (self.k_1 * dq + self.k_2 * ((v * dq - q * dv) / torch.pow(v, 2)) + self.k_3 * dv)
            ).reshape(shape)
        else:
            return (-p["V_0"] * ((self.k_1 + self.k_2) * dq + (self.k_3 - self.k_2) * dv)).reshape(
                shape
            )

    def hDavis(
        self,
        f: torch.Tensor = None,
        m: torch.Tensor = None,
        params: dict = None,
    ) -> torch.Tensor:
        """Compute the Davis (1998) BOLD signal.

        Parameters
        ----------
        f : torch.Tensor, optional
            Normalised cerebral blood flow.  Defaults to ``self.f``.
        m : torch.Tensor, optional
            Normalised CMRO₂.  Defaults to ``self.m``.
        params : dict, optional
            Override any key in :attr:`Davis_params`.

        Returns
        -------
        torch.Tensor
            BOLD signal: ``A * (1 - f^(α-β) * m^β)``.
        """
        if f is None:
            f = self.f
        if m is None:
            m = self.m

        p = {**self.Davis_params, **(params or {})}
        alpha = p["alpha"] if p["alpha"] is not None else 0.4

        f_safe = f.clamp(min=1e-8)
        m_safe = m.clamp(min=1e-8)
        exp_fm = alpha - p["beta"]
        return p["A"] * (
            1 - torch.exp(torch.log(f_safe) * exp_fm) * torch.exp(torch.log(m_safe) * p["beta"])
        )

    def dhDavis(
        self,
        f: torch.Tensor = None,
        m: torch.Tensor = None,
        df: torch.Tensor = None,
        dm: torch.Tensor = None,
        t: torch.Tensor = None,
        params: dict = None,
    ) -> torch.Tensor:
        """Compute the time derivative of the Davis (1998) BOLD signal.

        Parameters
        ----------
        f : torch.Tensor, optional
            Normalised CBF.  Defaults to ``self.f``.
        m : torch.Tensor, optional
            Normalised CMRO₂.  Defaults to ``self.m``.
        df : torch.Tensor, optional
            d*f*/d*t*.  Computed via autograd if ``None``.
        dm : torch.Tensor, optional
            d*m*/d*t*.  Computed via autograd if ``None``.
        t : torch.Tensor, optional
            Time tensor used for autograd when ``df`` or ``dm`` is ``None``.
        params : dict, optional
            Override any key in :attr:`Davis_params`.

        Returns
        -------
        torch.Tensor
            Time derivative dh/dt of the Davis BOLD signal.
        """
        p = {**self.Davis_params, **(params or {})}

        if f is None:
            f = self.f.squeeze() if self.f.ndim >= 2 else self.f
        if m is None:
            m = self.m.squeeze() if self.m.ndim >= 2 else self.m
        alpha = p["alpha"] if p["alpha"] is not None else 0.4

        if df is None:
            df = dfdt(signal=f, arg=t)
        if dm is None:
            dm = dfdt(signal=m, arg=t)

        # Analytical derivative: dh/dt = A * [(α-β)*f^(α-β-1)*m^β*df + β*f^(α-β)*m^(β-1)*dm]
        dhdt = p["A"] * (
            (alpha - p["beta"]) * df * f ** (alpha - p["beta"] - 1) * m ** p["beta"]
            + f ** (alpha - p["beta"]) * p["beta"] * dm * m ** (p["beta"] - 1)
        )
        return dhdt
