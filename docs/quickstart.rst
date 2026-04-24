Quick Start
===========

This guide walks through a minimal example of training a Balloon-PINN model.

Setup
-----

.. code-block:: python

   import torch
   from balloonlib.balloonpinnlib import (
       Multihead,
       loss,
       train,
       experimental_stims,
       segmentData,
       normFn,
   )

Building the model
------------------

.. code-block:: python

   model = Multihead(
       n_hidden=64,
       n_layers=4,
       impulse=True,
       dtype=torch.float32,
   )

Training
--------

.. code-block:: python

   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

   loss_trace = train(
       model=model,
       optimizer=optimizer,
       lossfn=loss,
       num_iter=5000,
       Balloon_params=balloon_params,
       data_params=data_params,
       domain=(0, 30),
   )

For a complete working example, see the notebooks in ``examples/``.
