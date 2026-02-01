svmix
=====

Ensemble stochastic volatility filter with Bayesian model averaging.

svmix implements a robust volatility estimation framework that maintains an
ensemble of particle filters, each with different model parameters. The library
automatically weights these filters using marginal likelihood, providing
uncertainty quantification and protection against parameter misspecification.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api/index
   theory

Installation
------------

Build the shared library and install the Python package:

.. code-block:: bash

   make pylib
   cd python && pip install -e .

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from svmix import Svmix, SvmixConfig, SvParams, Spec

   # Define parameter grid
   params = SvParams.grid(
       mu=np.linspace(-1.0, 0.0, 3),
       phi=np.linspace(0.95, 0.99, 3),
       sigma=np.linspace(0.1, 0.3, 3),
       nu=[5.0, 10.0],
   )

   # Initialize filter
   config = SvmixConfig(
       spec=Spec.VOL,
       num_models=len(params),
       num_particles=500
   )
   sv = Svmix(config, params)

   # Process observations
   returns = np.random.randn(100) * 0.02
   for r in returns:
       sv.step(r)

   # Extract posterior
   belief = sv.get_belief()


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
