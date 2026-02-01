Quick Start
===========

This guide covers the essential workflow for using svmix to estimate
stochastic volatility from return data.

Basic Usage
-----------

The svmix workflow consists of three steps:

1. Define a parameter grid
2. Initialize the filter
3. Process observations sequentially

.. code-block:: python

   import numpy as np
   from svmix import Svmix, SvmixConfig, SvParams, Spec

   # Step 1: Define parameter grid
   params = SvParams.grid(
       mu=np.linspace(-1.0, 0.0, 5),
       phi=np.linspace(0.95, 0.99, 5),
       sigma=np.linspace(0.1, 0.3, 5),
       nu=[5.0, 10.0, 20.0],
   )
   print(f"Grid size: {len(params)} parameter combinations")

   # Step 2: Initialize filter
   config = SvmixConfig(
       spec=Spec.VOL,
       num_models=len(params),
       num_particles=500
   )
   sv = Svmix(config, params)

   # Step 3: Process observations
   returns = np.random.randn(252) * 0.02  # Simulated daily returns
   for r in returns:
       sv.step(r)

   # Get results
   belief = sv.get_belief()
   print(f"Estimated volatility: {belief.vol:.4f}")

Understanding Belief
--------------------

The ``Belief`` object contains the posterior estimates:

.. code-block:: python

   belief = sv.get_belief()

   # Model-averaged volatility estimate
   print(f"Volatility: {belief.vol:.4f}")

   # Model-averaged log-volatility
   print(f"Log-volatility: {belief.log_vol:.4f}")

   # Model-averaged parameters
   print(f"mu: {belief.mu:.4f}")
   print(f"phi: {belief.phi:.4f}")
   print(f"sigma: {belief.sigma:.4f}")
   print(f"nu: {belief.nu:.4f}")

The volatility estimate is computed as a weighted average across all
models, where weights reflect each model's posterior probability given
the observed data.

Model Weights
-------------

Access the posterior model weights to understand which parameter
combinations best explain the data:

.. code-block:: python

   weights = sv.get_weights()

   # Find best model
   best_idx = np.argmax(weights)
   print(f"Best model: {params[best_idx]}")
   print(f"Weight: {weights[best_idx]:.4f}")

   # Effective number of models
   eff_k = sv.effective_num_models()
   print(f"Effective models: {eff_k:.1f} / {len(params)}")

A low effective number of models indicates strong evidence for a
particular parameter region.

Checkpointing
-------------

Save and restore filter state for long-running analyses:

.. code-block:: python

   # Save state
   sv.save_checkpoint("checkpoint.svmix")

   # Later: restore from checkpoint
   sv2 = Svmix.load_checkpoint("checkpoint.svmix")

   # Continue processing
   for r in new_returns:
       sv2.step(r)

Configuration Options
---------------------

The ``SvmixConfig`` class controls filter behavior:

.. code-block:: python

   config = SvmixConfig(
       spec=Spec.VOL,
       num_models=50,
       num_particles=1000,   # Particles per model
       lambda_=0.999,        # Exponential forgetting (default: 1.0)
       seed=42,              # Reproducibility
   )

- **spec**: Model specification. Currently only Spec.VOL is supported.

- **num_models**: Number of models in the ensemble. Should match the
  length of your parameter list.

- **num_particles**: More particles improve accuracy but increase
  computation. Start with 500, increase for production.

- **lambda_**: Exponential forgetting factor (0, 1]. Values less than
  1.0 apply exponential decay to historical likelihood contributions,
  allowing adaptation to regime changes. Default: 1.0 (no forgetting).

- **beta**: Softmax temperature (> 0). Controls weight sharpening.
  Lower values concentrate weights on best models. Typical: 0.5-1.0.

- **epsilon**: Anti-starvation weight floor [0, 1). Mixes uniform
  distribution into weights to prevent model collapse. Typical: 0.02-0.10.

- **seed**: Random seed for reproducible results. 0 for random.

Parameter Grid Design
---------------------

Careful grid design balances coverage with computational cost:

.. code-block:: python

   # Conservative grid for typical equity data
   params = SvParams.grid(
       mu=np.linspace(-1.5, 0.5, 5),     # Log-vol mean
       phi=np.linspace(0.90, 0.995, 5),  # Persistence
       sigma=np.linspace(0.05, 0.4, 5),  # Vol-of-vol
       nu=[4.0, 6.0, 10.0, 20.0],        # Tail weight
   )

Guidelines:

- **mu**: Range should cover plausible unconditional volatility levels.
  For daily returns, -2.0 to 0.5 covers most cases.

- **phi**: High persistence (0.95+) is typical for financial volatility.
  Include lower values if regime changes are expected.

- **sigma**: Controls volatility of log-volatility. Higher values
  produce more volatile volatility estimates.

- **nu**: Lower values produce heavier tails. Include a range to
  capture different tail behaviors.

Memory Management
-----------------

The filter allocates significant memory. For long-running processes,
explicitly free resources:

.. code-block:: python

   sv = Svmix(params, config)
   try:
       for r in returns:
           sv.step(r)
   finally:
       sv.free()

Or use a context manager pattern:

.. code-block:: python

   sv = Svmix(params, config)
   try:
       # ... processing ...
       pass
   finally:
       sv.free()

Thread Safety
-------------

Individual ``Svmix`` instances are not thread-safe. For parallel
processing, create separate instances per thread. The underlying
C library uses OpenMP for intra-filter parallelization.
