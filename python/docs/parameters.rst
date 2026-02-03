Parameter Guide
===============

This guide explains the meaning and interpretation of all input parameters
and output variables in svmix.

Model Parameters
----------------

The stochastic volatility model has four core parameters that control its
behavior. These are specified in the :class:`~svmix.params.SvParams` class.

.. warning::

   **Parameter ranges are asset-specific**: The typical values and ranges
   provided below are educated estimates based on common equity data.
   Actual optimal values vary significantly by:
   
   - Asset class (equities, crypto, FX, commodities)
   - Time resolution (tick, 1-min, daily, weekly)
   - Market regime (calm vs crisis)
   - Individual asset characteristics
   
   Always start with wide grids and examine posterior model weights to
   determine appropriate ranges for your specific use case.

Persistence: φ (phi)
^^^^^^^^^^^^^^^^^^^^

**Definition**: AR(1) coefficient in the log-volatility process.

**Range**: (0, 1)

**Equation**: :math:`h_t = \mu + \phi(h_{t-1} - \mu) + \sigma\eta_t`

**Interpretation**:

- **φ = 0.5**: Low persistence - volatility shocks dissipate quickly
  (half-life of 1 period)
  
- **φ = 0.9**: Moderate persistence - typical for many assets
  (half-life ≈ 7 periods)
  
- **φ = 0.98**: High persistence - volatility clusters strongly
  (half-life ≈ 35 periods)

**Financial Intuition**: Higher φ means volatility tends to remain elevated
for longer periods after a shock. This captures the well-known "volatility
clustering" in financial markets - turbulent times tend to be followed by
more turbulent times.

**Typical Values**:

- Daily equity returns: 0.95-0.99
- Intraday (1-min) returns: 0.90-0.97
- Weekly/monthly: 0.85-0.95

*Note: These are rough guidelines. Individual assets can vary significantly.*

**Grid Design**: Use a narrow range around 0.97 for daily equity data. For
unknown assets, span 0.90-0.99 to be safe.

Volatility of Volatility: σ (sigma)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: Standard deviation of innovations to log-volatility.

**Range**: (0, ∞), typically (0.05, 0.5)

**Equation**: Innovation variance is :math:`\sigma^2`

**Interpretation**:

- **σ = 0.1**: Stable volatility - changes gradually
- **σ = 0.2**: Moderate variability - typical for equity indices
- **σ = 0.4**: Highly volatile volatility - common in crypto, individual stocks

**Financial Intuition**: This controls how quickly volatility can change.
Higher σ allows for rapid shifts between calm and turbulent periods. Lower
σ produces smoother volatility estimates.

**Typical Values**:

- Major equity indices (SPY, QQQ): 0.15-0.25
- Individual stocks: 0.20-0.35
- Cryptocurrencies: 0.25-0.45

*Note: Highly dependent on market conditions and specific asset behavior.*

**Grid Design**: For equity indices, try 0.10-0.30. For crypto, expand to
0.15-0.45.

**Warning**: Very low σ (< 0.05) can cause the filter to react too slowly
to regime changes. Very high σ (> 0.5) can produce noisy estimates.

Degrees of Freedom: ν (nu)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: Degrees of freedom for Student-t observation distribution.

**Range**: (2, ∞), typically (4, 30)

**Interpretation**:

- **ν = 4**: Very heavy tails - allows for extreme outliers
  (kurtosis ≈ ∞ for ν ≤ 4)
  
- **ν = 10**: Moderate tails - typical for equity returns
  (kurtosis = 4.0)
  
- **ν = 30**: Nearly Gaussian tails
  (kurtosis ≈ 3.2)

**Financial Intuition**: Lower ν assigns higher probability to large moves
(crashes, flash crashes). This helps the filter not overreact to extreme
events, treating them as expected tail events rather than regime changes.

**Typical Values**:

- Equity indices: 8-15
- Individual stocks: 5-10
- Cryptocurrencies: 4-8
- High-frequency data: 4-6 (more outliers)

*Note: Tail behavior varies with asset liquidity, market structure, and
data quality. Use wider grids for unknown assets.*

**Grid Design**: For robustness, include both heavy (4-6) and moderate
(8-20) tail options. The filter will automatically select the appropriate
tail weight based on observed data.

**Technical Note**: ν must be > 2 for finite variance. Values ≤ 2 are not
supported.

Long-Run Mean: μ (mu)
^^^^^^^^^^^^^^^^^^^^^^

**Definition**: Unconditional mean of log-volatility.

**Range**: (-∞, ∞), typically (-2, 0)

**Equation**: :math:`E[h_t] = \mu`

**Interpretation**:

- **μ = -10.0**: Unconditional volatility ≈ 0.67% daily (10.7% annualized)
- **μ = -9.0**: Unconditional volatility ≈ 1.11% daily (17.7% annualized)
- **μ = -8.0**: Unconditional volatility ≈ 1.83% daily (29.1% annualized)

**Conversion**: :math:`\text{volatility} = \exp(\mu / 2)` (in return units)

**Financial Intuition**: This anchors where volatility "wants to be" in
the long run. After shocks, volatility mean-reverts toward
:math:`\exp(\mu/2)`.

**Typical Values**:

- Major equity indices: -10.0 to -8.5 (≈ 10-20% annualized)
- Individual stocks: -9.0 to -7.5 (≈ 15-30% annualized)
- Cryptocurrencies: -8.0 to -6.0 (≈ 30-100% annualized)

*Note: These ranges assume normal market conditions. Crisis periods,
launches, or regime changes can produce values far outside these ranges.*

**Grid Design**: For unknown assets, use a wide range (-10, -7). The
filter will identify the appropriate level from data.

**Practical Tip**: If you have a rough idea of average volatility, convert
it to log-space:

.. code-block:: python

   avg_volatility = 0.015  # 1.5% daily
   mu = 2 * np.log(avg_volatility)  # ≈ -8.47

Configuration Parameters
------------------------

The :class:`~svmix.config.SvmixConfig` class controls the filter's behavior.

Number of Models: K
^^^^^^^^^^^^^^^^^^^

**Definition**: Size of the model ensemble.

**Range**: [1, ∞), typically 20-150

**Interpretation**: More models provide better coverage of parameter space
but increase computation linearly.

**Tradeoffs**:

- **K = 20**: Fast, suitable for real-time applications, coarse parameter grid
- **K = 50**: Balanced for production
- **K = 100**: High accuracy, offline analysis
- **K = 200+**: Diminishing returns unless parameter space is high-dimensional

**Rule of Thumb**: Use K such that the parameter grid is fine enough to
capture expected variation. For 1D grids (varying only φ), K=30 is often
sufficient. For full 4D grids, K=100+ may be needed.

Number of Particles: N
^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: Particles per particle filter (per model).

**Range**: [50, ∞), typically 250-1000

**Interpretation**: More particles reduce Monte Carlo error in each filter.

**Tradeoffs**:

- **N = 100**: Fast but noisy estimates
- **N = 500**: Good balance for production
- **N = 1000**: High precision, slower
- **N = 2000+**: Diminishing returns

**Computational Cost**: Total cost scales as O(K × N).

**Recommendation**: Start with N=500. If volatility estimates are noisy,
increase to 1000. If speed is critical, reduce to 250.

Forgetting Factor: λ
^^^^^^^^^^^^^^^^^^^^

**Definition**: Exponential decay applied to historical likelihood contributions.

**Range**: (0, 1], default: 1.0 (no forgetting)

**Equation**: Log-likelihood is weighted as :math:`\lambda^{T-t} \log p(r_t)`

**Interpretation**:

- **λ = 1.0**: All observations weighted equally (stationary data)
- **λ = 0.999**: Recent data 2× more important than data 693 periods ago
- **λ = 0.99**: Recent data 2× more important than data 69 periods ago

**Use Cases**:

- **Stationary markets**: λ = 1.0 (default)
- **Suspected regime changes**: λ = 0.995-0.999
- **Non-stationary data**: λ = 0.99-0.995

**Warning**: Lower λ allows faster adaptation but reduces effective sample
size for parameter estimation. Can lead to unstable weights if data is
actually stationary.

Softmax Temperature: β
^^^^^^^^^^^^^^^^^^^^^^

**Definition**: Controls sharpening of model weights.

**Range**: (0, ∞), typically 0.5-1.5

**Effect**:

- **β → 0**: Weights concentrate on single best model (greedy)
- **β = 1.0**: Standard softmax (default)
- **β > 1**: Weights more diffuse (conservative)

**Interpretation**: Lower β gives more aggressive model selection. Higher
β hedges more across models.

**Typical Values**: 0.5-1.0 for most applications. Use 0.8 as default.

Anti-Starvation: ε
^^^^^^^^^^^^^^^^^^

**Definition**: Minimum weight floor to prevent model collapse.

**Range**: [0, 1), typically 0.02-0.10

**Equation**: Weight update includes uniform mixing:
:math:`w_t^{(k)} \leftarrow (1-\epsilon)w_t^{(k)} + \epsilon/K`

**Interpretation**:

- **ε = 0**: No protection (pure Bayesian updating)
- **ε = 0.05**: Each model maintains at least 5%/K base probability
- **ε = 0.10**: Conservative - hedges heavily across models

**Use Cases**:

- **Stable data**: ε = 0.02 (minimal hedging)
- **Regime changes expected**: ε = 0.05-0.10 (more hedging)

**Warning**: Higher ε slows convergence to the true model. Lower ε risks
premature model collapse.

Output: Belief Structure
------------------------

The :class:`~svmix.types.Belief` object contains the filter's posterior estimates.

Volatility: vol
^^^^^^^^^^^^^^^

**Definition**: Model-averaged instantaneous volatility estimate.

**Units**: Same as input returns (e.g., if returns are daily, vol is daily)

**Interpretation**: The filter's best guess for current volatility. This is
the standard deviation of returns conditional on current information.

**Usage**:

.. code-block:: python

   belief = svmix.get_belief()
   daily_vol = belief.vol
   annual_vol = daily_vol * np.sqrt(252)  # Annualize
   
   # 95% prediction interval for next return
   lower = -1.96 * daily_vol
   upper = +1.96 * daily_vol

**Note**: This is computed as :math:`\exp(\text{mean\_h}/2)` where mean_h
is the model-averaged log-volatility: :math:`\text{mean\_h} = \sum_k w_k h_t^{(k)}`.
Due to Jensen's inequality, this differs slightly from
:math:`\sum_k w_k \exp(h_t^{(k)}/2)`.

Log-Volatility: log_vol
^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: Mean of log-volatility in log-space.

**Units**: Natural log of volatility

**Interpretation**: :math:`E[h_t]` under the posterior. This is NOT
:math:`\log(\text{vol})` due to Jensen's inequality.

**Usage**: Primarily for internal diagnostics. Use ``vol`` for trading applications.

Mean Parameters: mu, phi, sigma, nu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: Model-averaged parameter estimates.

**Interpretation**: The weighted average of parameters across all models:

.. math::

   \bar{\theta} = \sum_{k=1}^K w_k \theta_k

where :math:`w_k` are the model posterior probabilities.

**Usage**:

.. code-block:: python

   belief = svmix.get_belief()
   print(f"Estimated volatility: {belief.vol:.4f}")
   print(f"Uncertainty (var_h): {belief.var_h:.4f}")

Weighted Parameters
^^^^^^^^^^^^^^^^^^^

**Access**: Use :meth:`~svmix.Svmix.get_weighted_params` to get Bayesian Model Average (BMA) parameter estimates.

**Definition**: The weighted average of all model parameters using current posterior probabilities:

.. math::

   \bar{\theta} = \sum_{k=1}^K w_k \theta_k

where :math:`w_k` are the model weights from :meth:`~svmix.Svmix.get_weights`.

**Returns**: Dictionary with keys ``phi``, ``sigma``, ``nu``, ``mu``.

**Usage**:

.. code-block:: python

   weighted = svmix.get_weighted_params()
   
   print(f"Effective persistence: {weighted['phi']:.3f}")
   print(f"Effective vol-of-vol: {weighted['sigma']:.3f}")
   print(f"Effective tail heaviness: {weighted['nu']:.1f}")
   print(f"Effective long-run mean: {weighted['mu']:.3f}")

**Interpretation**:

- **weighted['phi']**: Current persistence regime (high → shocks last longer)
- **weighted['sigma']**: Current vol-of-vol regime (high → more volatility changes)
- **weighted['nu']**: Current tail regime (low → expect extreme events)
- **weighted['mu']**: Current long-run vol level

**Applications**: Feature extraction for ML trading algorithms, regime detection,
risk management. See :doc:`ml_features` for detailed guidance.

Weight Entropy
^^^^^^^^^^^^^^

**Access**: Use :meth:`~svmix.Svmix.get_weight_entropy` to measure regime ambiguity.

**Definition**: Shannon entropy of model weights:

.. math::

   H = -\sum_{k=1}^K w_k \log(w_k)

**Returns**: Float in range [0, log(K)] where K is number of models.

**Interpretation**:

- **Low entropy** (→ 0): One model dominates, clear regime identification
- **High entropy** (→ log K): Models have similar weights, regime unclear

**Usage**:

.. code-block:: python

   import math
   
   entropy_bits = svmix.get_weight_entropy(base=2.0)  # bits
   entropy_nats = svmix.get_weight_entropy(base=math.e)  # nats
   
   max_entropy = math.log2(config.num_models)
   normalized = entropy_bits / max_entropy  # [0, 1]
   
   if normalized > 0.8:
       print("High regime ambiguity - multiple models plausible")
   else:
       print("Clear regime - one model dominates")

**Applications**: Confidence indicator for volatility estimates, regime transition
detector, signal quality metric for trading algorithms.

Mean Log-Volatility: mean_h
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: Posterior mean of :math:`h_t`.

**Relationship**: ``mean_h`` ≈ ``2 * log(vol)`` but not exactly due to
averaging across models.

**Usage**: Primarily diagnostic. Prefer ``vol`` for interpretability.

Log-Volatility Variance: var_h
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: Posterior variance of :math:`h_t`.

**Interpretation**: Uncertainty in the log-volatility estimate. High
variance indicates:

- Early in filtering (insufficient data)
- Regime change (models disagree)
- Poor parameter specification

**Usage**:

.. code-block:: python

   belief = svmix.get_belief()
   if belief.var_h > 0.5:
       print("High uncertainty - be cautious")

Mean Volatility of Volatility: mean_sigma
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Definition**: Posterior mean of σ parameter across models.

**Usage**: Diagnostic. Reveals the effective volatility-of-volatility the
ensemble has learned.

Valid Flag: valid
^^^^^^^^^^^^^^^^^

**Definition**: Boolean indicating if the belief is valid.

**False when**:

- No observations processed yet (t=0)
- Filter encountered numerical issues
- All particle weights collapsed

**Usage**:

.. code-block:: python

   belief = svmix.get_belief()
   if not belief.valid:
       print("Warning: Invalid belief state")

Practical Examples
------------------

The following examples demonstrate parameter selection for different use
cases. **These are starting points, not definitive specifications**.
Always validate parameter choices by:

1. Examining posterior model weights after filtering
2. Checking if effective_num_models() is reasonable (not collapsed to 1-2)
3. Comparing volatility estimates against realized volatility
4. Testing on out-of-sample data

Example 1: Daily Equity Returns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Objective**: Estimate volatility for SPY (S&P 500 ETF).

**Parameter choices**:

.. code-block:: python

   params = SvParams.grid(
       phi=[0.96, 0.97, 0.98, 0.99],     # High persistence
       sigma=[0.15, 0.20, 0.25],         # Moderate vol-of-vol
       nu=[6, 10, 15],                   # Moderate to light tails
       mu=np.linspace(-10.0, -8.5, 5)    # 10-20% annualized
   )

**Reasoning**: SPY has stable, highly persistent volatility with moderate
tail risk.

Example 2: Cryptocurrency Returns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Objective**: Estimate volatility for Bitcoin.

**Parameter choices**:

.. code-block:: python

   params = SvParams.grid(
       phi=[0.92, 0.95, 0.97, 0.99],     # Wider persistence range
       sigma=[0.25, 0.35, 0.45],         # High vol-of-vol
       nu=[4, 6, 8, 12],                 # Heavy tails
       mu=np.linspace(-8.0, -6.0, 7)     # 30-100% annualized
   )

**Reasoning**: Crypto exhibits rapid volatility shifts, extreme events, and
higher baseline volatility.

Example 3: High-Frequency Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Objective**: 1-minute returns on futures.

**Parameter choices**:

.. code-block:: python

   params = SvParams.grid(
       phi=[0.90, 0.93, 0.96, 0.98],     # Lower persistence (faster decay)
       sigma=[0.20, 0.30, 0.40],         # Rapid changes
       nu=[4, 5, 7, 10],                 # Heavy tails (microstructure noise)
       mu=np.linspace(-12.0, -10.0, 6)   # Very low per-minute volatility
   )

**Reasoning**: Intraday volatility mean-reverts faster, with frequent noise
spikes requiring heavy tails.

Common Pitfalls
---------------

**Pitfall 1**: Using simple returns instead of log returns

See :doc:`data_requirements` for why this breaks the model.

**Pitfall 2**: Too narrow parameter grid

If the true parameters lie outside your grid, the filter will select the
best available option but performance will degrade.

**Solution**: Start with a wide grid, examine converged weights, then
refine.

**Pitfall 3**: Interpreting μ as volatility

μ is the log-space mean. To get volatility level: ``np.exp(mu / 2)``.

**Pitfall 4**: Forgetting to annualize

If your returns are daily, ``belief.vol`` is daily volatility. To
annualize: ``belief.vol * np.sqrt(252)``.

**Pitfall 5**: Not monitoring effective_num_models()

If this drops to 1-2, your grid may be too coarse or misspecified.

Further Reading
---------------

- Model specification details: :doc:`theory`
- Complete API reference: :doc:`api/index`
- Data preprocessing: :doc:`data_requirements`
