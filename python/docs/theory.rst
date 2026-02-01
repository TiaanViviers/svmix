Theory
======

This page describes the mathematical foundations of the svmix library.

Stochastic Volatility Model
---------------------------

The standard stochastic volatility model specifies returns as:

.. math::

   r_t = \exp(h_t / 2) \, \epsilon_t

where :math:`h_t` is the log-volatility process following an AR(1) dynamic:

.. math::

   h_t = \mu + \phi (h_{t-1} - \mu) + \sigma \eta_t

The innovations :math:`\epsilon_t` and :math:`\eta_t` are independent
standard normal random variables.

Student-t Observations
----------------------

To capture the heavy tails observed in financial returns, svmix uses
Student-t distributed observations:

.. math::

   r_t \sim \text{Student-t}(\nu, 0, \exp(h_t / 2))

The degrees of freedom parameter :math:`\nu` controls tail weight:

- :math:`\nu \to \infty`: Gaussian tails
- :math:`\nu = 4`: Heavy tails with finite kurtosis
- :math:`\nu \leq 4`: Extremely heavy tails

Model Parameters
----------------

The complete parameter vector :math:`\theta = (\mu, \phi, \sigma, \nu)`:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
     - Typical Range
   * - :math:`\mu`
     - Unconditional mean of log-volatility
     - [-2.0, 0.5]
   * - :math:`\phi`
     - Persistence (AR(1) coefficient)
     - [0.90, 0.999]
   * - :math:`\sigma`
     - Volatility of log-volatility
     - [0.05, 0.4]
   * - :math:`\nu`
     - Degrees of freedom
     - [4, 30]

Particle Filtering
------------------

For a given parameter :math:`\theta`, the particle filter approximates
the filtering distribution:

.. math::

   p(h_t | r_{1:t}, \theta) \approx \sum_{i=1}^N w_t^{(i)} \delta_{h_t^{(i)}}(h_t)

where :math:`\{h_t^{(i)}, w_t^{(i)}\}_{i=1}^N` are weighted particles.

At each time step, the filter:

1. **Propagates** particles through the state transition
2. **Weights** particles by observation likelihood
3. **Resamples** to prevent weight degeneracy

The marginal likelihood increment is:

.. math::

   p(r_t | r_{1:t-1}, \theta) \approx \frac{1}{N} \sum_{i=1}^N w_t^{(i)}

Bayesian Model Averaging
------------------------

svmix maintains an ensemble of :math:`K` particle filters, each with
different parameters :math:`\theta_k`. The posterior model probability is:

.. math::

   P(\theta_k | r_{1:t}) \propto P(\theta_k | r_{1:t-1}) \cdot p(r_t | r_{1:t-1}, \theta_k)

Starting from uniform priors, the weights evolve as:

.. math::

   w_t^{(k)} \propto w_{t-1}^{(k)} \cdot \hat{p}(r_t | r_{1:t-1}, \theta_k)

where :math:`\hat{p}(r_t | r_{1:t-1}, \theta_k)` is the marginal
likelihood estimate from filter :math:`k`.

The model-averaged volatility estimate is:

.. math::

   \hat{h}_t = \sum_{k=1}^K w_t^{(k)} \, \mathbb{E}[h_t | r_{1:t}, \theta_k]

This Bayesian approach provides:

- **Robustness** to parameter uncertainty
- **Automatic** model selection as data accumulates
- **Uncertainty quantification** through weight dispersion

Forgetting Factor
-----------------

For non-stationary data, exponential forgetting downweights old
observations:

.. math::

   w_t^{(k)} \propto w_{t-1}^{(k) \lambda} \cdot p(r_t | r_{1:t-1}, \theta_k)

where :math:`\lambda \in (0, 1]` is the forgetting factor. This allows
the ensemble to adapt to regime changes.

Anti-Starvation Mixing
----------------------

To prevent premature model collapse, svmix mixes a small uniform
component into the weights at each step:

.. math::

   w_t^{(k)} \leftarrow (1 - \alpha) w_t^{(k)} + \frac{\alpha}{K}

This ensures all models retain some probability mass, allowing
recovery if the data distribution shifts.

Effective Number of Models
--------------------------

The effective number of models quantifies weight concentration:

.. math::

   K_{\text{eff}} = \exp\left( -\sum_{k=1}^K w_t^{(k)} \log w_t^{(k)} \right)

- :math:`K_{\text{eff}} = K`: Uniform weights (maximum uncertainty)
- :math:`K_{\text{eff}} = 1`: Single dominant model (point estimate)

Monitoring :math:`K_{\text{eff}}` helps detect when the ensemble has
converged to a particular parameter region.

References
----------

- Kim, S., Shephard, N., & Chib, S. (1998). Stochastic volatility:
  Likelihood inference and comparison with ARCH models.
  *Review of Economic Studies*, 65(3), 361-393.

- Doucet, A., de Freitas, N., & Gordon, N. (2001).
  *Sequential Monte Carlo methods in practice*. Springer.

- Liu, J. S. (2001). *Monte Carlo strategies in scientific computing*.
  Springer.
