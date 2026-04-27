Quick Start
===========

End-to-end demonstration of BalloonLib: from numerical simulation of
haemodynamics to PINN training and evaluation.

**Pipeline**

.. code-block:: text

   Stimulus → Numerical ODE (balloonmodellib)  →  reference f, m, v, q, BOLD
            → PINN (Multihead)                →  learned f, m, v, q, HRF
            → Evaluation (kge_stat, hrf_description)
            → [Optional] Noisy BOLD training

.. note::

   GPU is recommended. The library detects CUDA automatically.

A fully executable version of this guide is available as
``examples/QuickStart.ipynb``.


0. Setup and imports
--------------------

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   import numpy as np
   import matplotlib.pyplot as plt

   from balloonlib import balloonmodellib as bml
   from balloonlib.model    import Multihead
   from balloonlib.training import loss, train
   from balloonlib.data     import experimental_stims, segmentData, normFn
   from balloonlib.utils    import np2tensor, tensor2np, tofit, timeBall
   from balloonlib.metrics  import kge_stat, hrf_description
   from balloonlib.plotting import plot_balloon_fitting, plot_trace, plot_weights

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   torch.set_default_device(device)
   torch.set_default_dtype(torch.float32)
   print(f'Using device: {device}')


1. Stimulus design
------------------

We use a block-design stimulus with 10 onsets at 1.75 s TR (166 volumes, ~4.8 min).

.. code-block:: python

   # fMRI acquisition parameters
   TR        = 1.75          # Repetition time (s)
   n_samples = 166           # number of fMRI volumes
   Bold_time = np.arange(0, n_samples * TR, TR)

   sti_len    = 3            # stimulus block duration (s)
   Sti_Onsets = np.array([22, 42, 79, 99, 134, 163, 191, 209, 236, 269]) - 1

   # High-res (0.01 s) impulse used by the PINN — 30 s at 100 Hz
   I = torch.zeros(30).repeat(100)
   I[5:105] = 1   # stimulus on from 0.05 s to 1.05 s

   # Overall stimulus for BOLD convolution
   Overall_stims, stim_time = experimental_stims(
       normDataSize = n_samples,
       Sti_Onsets   = Sti_Onsets,
       TR           = TR,
       block_len    = sti_len,
       stmxblck     = 1,
       Hz           = 100,
   )
   stim_time_np = tensor2np(stim_time)


2. Numerical Balloon model (reference solution)
------------------------------------------------

:mod:`balloonlib.balloonmodellib` solves the Balloon ODE system numerically
using scipy. These solutions serve as ground-truth overlays when evaluating
the PINN.

.. code-block:: python

   f_parameters = {'gamma': 1/2.46, 'kappa': 1/1.54}

   # Neurovascular coupling → CBF (f) and CMRO₂ (m)
   f_np, _ = bml.NeurovascularCoupling(
       stimulus=tensor2np(I), version='differential',
       params=f_parameters, AmpI=0.2,
   )
   m_np, _ = bml.NeurovascularCoupling(
       stimulus=tensor2np(I), version='differential',
       params=f_parameters, AmpI=0.05,
   )

   # Balloon core → blood volume (v) and deoxyhaemoglobin (q)
   vol_q_parameters = {'tau_MTT': 3.0, 'alpha': 0.4, 'tau_m': 20}
   v_np, q_np = bml.Balloon_odeint(f_np, m_np, params=vol_q_parameters, viscoelastic=True)

   # BOLD signal
   bold_params = {'E_0': 0.32, 'V_0': 0.03, 'TE': 0.06, 'O_o': 40.3, 'r_0': 25, 'eps': 1.43}
   bold_np     = bml.BOLD_func(v_np, q_np, params=bold_params, BM='classic')

   # Convert to tensors
   f_t = np2tensor(f_np);  m_t = np2tensor(m_np)
   v_t = np2tensor(v_np);  q_t = np2tensor(q_np)
   num_balloon_hrf = np2tensor(bold_np)


3. Simulate BOLD signal
-----------------------

Convolve the numerical HRF with the experimental stimulus to get a synthetic
BOLD signal, then subsample at the fMRI TR.

.. code-block:: python

   sampling_rate = int(TR * 100)   # samples to skip = TR × Hz

   num_bold_test, num_bold_time = tofit(Overall_stims, num_balloon_hrf, n_samples * TR)
   sampled_bold_test = tensor2np(num_bold_test[::sampling_rate])


4. Configure training parameters
---------------------------------

Two dictionaries are passed to :func:`~balloonlib.training.train`:

- ``Balloon_params`` — physiological ODE constants.
- ``data_params``    — fMRI data, stimulus metadata, and training settings.

.. code-block:: python

   domain        = (0, len(I) * 0.01)   # physical time window: [0, 30] s
   num_iter      = 5000
   learning_rate = 4e-3
   percent, step_size = 0.85, 1000      # StepLR: lr × 0.85 every 1000 iters

   # Initial loss weights
   p     = 0.6
   w_raw = {
       'ode':    [p    ],   # ODE residual
       'bold':   [1 - p],   # data fit
       'ic':     [1.   ],   # Dirichlet IC  (f=m=v=q=1 at t=0)
       'border': [1.   ],   # Cauchy IC     (derivatives=0 at t=0)
       'other':  [0.   ],   # physics violation penalty (off by default)
   }

   data_dict = {
       'TR':                TR,
       'Sti_Onsets':        Sti_Onsets,
       'Bold_Signal':       sampled_bold_test,   # numpy array, shape (n_samples,)
       'errorFn':           nn.MSELoss(),
       'stim_length [seg]': sti_len,
       'stim_x_block':      1,
       't0':                0,
   }

   Balloon_dict = {
       'lambdar_list': [0.2, 0.05],       # [CBF amplitude, CMRO2 amplitude]
       'kappa_list':   [1/1.54, 1/1.54],  # [CBF decay, CMRO2 decay]  (Maith 2022)
       'gamma_list':   [1/2.46, 1/2.46],  # [CBF return, CMRO2 return]
       'tau_m_list':   20,                # viscoelastic time constant (s)
       'tau_MTT_list': 3.0,               # mean transit time (s)
       'alpha':        0.4,               # Grubb's stiffness exponent
       'I':            I,                 # high-res stimulus tensor
   }


5. Build the PINN
-----------------

:class:`~balloonlib.model.Multihead` simultaneously predicts all four Balloon
state variables (*f*, *m*, *v*, *q*) from a normalised time input.

.. code-block:: python

   model = Multihead(
       use_fourier          = False,   # set True for Fourier feature encoding
       fourier_mapping_size = 9,
       fourier_scale        = 0.85,
       fourier_learnable    = True,
       multi_scale_fourier  = False,
       random_weightsMatrix = True,    # use FactorizedLinear (RWF) layers
       impulse              = False,   # time-only input
       act                  = nn.SiLU(),
       nv_fn                = (nn.Softplus(), nn.Softplus()),
       core_fn              = (nn.Softplus(), nn.Softplus()),
       dtype                = torch.float32,
   )
   model.init_nn_params()

   total_params = sum(p.numel() for p in model.parameters())
   print(f'PINN ready: {total_params:,} trainable parameters')


6. Train
--------

:func:`~balloonlib.training.train` normalises time to zero-mean / unit-std,
segments the BOLD data into stimulus-locked epochs, and runs ``num_iter``
Adam steps with StepLR decay.

.. code-block:: python

   optimizer = optim.Adam(model.parameters(), lr=learning_rate)
   scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=percent)

   loss_trace = train(
       model          = model,
       optimizer      = optimizer,
       lossfn         = loss,
       num_iter       = num_iter,
       Balloon_params = Balloon_dict,
       data_params    = data_dict,
       domain         = domain,
       random         = False,
       every          = num_iter // 3,
       loss_weights   = w_raw,
       scheduler      = scheduler,
       dtype          = torch.float32,
   )


7. Visualise results
--------------------

.. code-block:: python

   # Normalised time axis
   t    = torch.arange(0, I.shape[0]) / I.shape[0]
   t_nn = ((t - t.mean()) / t.std()).view(-1, 1)

   # Single-trial stimulus for the BOLD panel
   stimulus, stimulus_time = experimental_stims(
       normDataSize = 13,
       Sti_Onsets   = [TR],
       TR           = TR,
       block_len    = sti_len,
       stmxblck     = 1,
   )

   plot_balloon_fitting(
       model               = model,
       t_normalized        = t_nn,
       domain              = domain,
       stimulus            = stimulus,
       title               = 'Noise-free simulation — PINN fit',
       numerical_solutions = {'f': f_t, 'm': m_t, 'v': v_t, 'q': q_t, 'bold': num_balloon_hrf},
       data_params         = {
           'TR':               TR,
           'Sti_Onsets':       Sti_Onsets,
           'Bold_Signal':      sampled_bold_test,
           'Bold_data_time':   num_bold_time[::sampling_rate],
           'stim_length [seg]': sti_len,
           'stimulus':          stimulus,
           'stimulus_time':     stimulus_time,
           'Overallstim':       Overall_stims,
           'Overall_stim_time': stim_time,
       },
       first_non_zero_index = torch.argmax(I) - 1,
       show_bold_signal     = True,
   )

   plot_trace(loss_trace, title='Loss — noise-free simulation', step_size=step_size)
   plot_weights(w_raw, title='Adaptive weights', keys_to_skip=[], step_size=step_size)


8. Quantitative evaluation
---------------------------

.. code-block:: python

   model.eval()
   with torch.no_grad():
       _, _ = model(t_nn)
       hrf_pinn = model.predictor()

   # Convolve HRF with overall stimulus → BOLD prediction
   bold_pred, bold_pred_time = tofit(Overall_stims, hrf_pinn, n_samples * TR)

   # Subsample to fMRI TR
   idx, _ = timeBall(torch.as_tensor(Bold_time), bold_pred_time)
   bold_at_TR = tensor2np(bold_pred[idx])

   # Kling-Gupta Efficiency (1.0 = perfect)
   kge = kge_stat(y_obs=sampled_bold_test, y_sim=bold_at_TR)
   print(f'KGE = {kge:.4f}  (1.0 = perfect)')

   # HRF shape descriptors
   desc = hrf_description(tensor2np(hrf_pinn), max_time=domain[1])
   for k, v_d in desc.items():
       print(f'  {k}: {v_d[0]:.4f}')


Next steps
----------

- **Real fMRI data** — replace ``sampled_bold_test`` with your normalised BOLD
  signal using :func:`~balloonlib.data.normFn`.
- **Adaptive reweighting** — call ``loss_reweight_paranoid()`` between training
  phases to adjust loss weights dynamically.
- **Fourier features** — set ``use_fourier=True`` in :class:`~balloonlib.model.Multihead`
  for improved high-frequency fitting.
- **Noisy BOLD** — see ``examples/QuickStart.ipynb`` sections 10-12 for a full
  demonstration of training on noisy simulations (Shan 2014 noise model).
- **Deeper pipeline walkthrough** — see ``examples/BalloonExample.ipynb`` for a
  detailed dive into the ``balloonmodellib`` numerical pipeline.