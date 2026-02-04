Feature Extraction
=====================

svmix provides efficient methods for extracting minimal, high-signal features. 
These features quantify regime state and model uncertainty without redundancy.

Core Feature Set
----------------

Six essential features capture the complete state:

1. **vol** - Current volatility estimate (annualized %)
2. **var_h** - Uncertainty in log-volatility
3. **weight_entropy** - Regime ambiguity
4. **weighted_phi** - Persistence regime
5. **weighted_nu** - Tail risk regime  
6. **log_likelihood** - Model fit quality

Complete Example
----------------

.. code-block:: python

   import numpy as np
   from svmix import Svmix, SvmixConfig, SvParams, Spec

   # Setup: 50-model ensemble spanning realistic parameter space
   config = SvmixConfig(
       spec=Spec.VOL,
       num_models=50,
       num_particles=500,
       lambda_=0.995,
       beta=0.8,
       epsilon=0.05
   )
   
   params = SvParams.linspace(
       50,
       phi=(0.90, 0.99),    # Low to high persistence
       sigma=0.2,           # Vol-of-vol
       nu=10.0,             # Degrees of freedom
       mu=-9.0              # ~1% daily vol
   )
   
   svmix = Svmix(config, params)
   
   # Process returns and extract features
   features_history = []
   
   for log_return in log_returns:
       svmix.step(log_return)
       
       belief = svmix.get_belief()
       weighted = svmix.get_weighted_params()
       
       features = {
           'vol': belief.vol,                        # Current volatility
           'var_h': belief.var_h,                    # Uncertainty
           'weight_entropy': svmix.get_weight_entropy(),  # Regime ambiguity
           'weighted_phi': weighted['phi'],          # Persistence
           'weighted_nu': weighted['nu'],            # Tail risk
           'log_likelihood': svmix.get_last_log_likelihood()  # Fit
       }
       features_history.append(features)
   
   svmix.free()

Feature Interpretation
----------------------

vol: Current Volatility
^^^^^^^^^^^^^^^^^^^^^^^^

Volatility estimate with configurable units.

**Default (belief.vol):** Annualized percentage (e.g., 18.2 = 18.2% annual)

**Flexible units via get_vol():**

.. code-block:: python

   # Annualized percentage (default) - standard finance convention
   vol = belief.get_vol(annualize=True, as_percentage=True)
   # or simply: vol = belief.vol
   # e.g., 18.2 means "18.2% annual volatility"
   
   # Daily percentage - matches your return scale
   vol = belief.get_vol(annualize=False, as_percentage=True)
   # e.g., 1.15 means "1.15% daily moves"
   
   # Daily decimal - for calculations
   vol = belief.get_vol(annualize=False, as_percentage=False)
   # e.g., 0.0115, use directly in position sizing
   
   # Annualized decimal - for Sharpe ratios
   vol = belief.get_vol(annualize=True, as_percentage=False)
   # e.g., 0.182

**Range:** 
- Annualized %: 0-200% (typical equity: 10-80%)
- Daily decimal: 0.0-2.0 (typical equity: 0.006-0.05)

**Usage examples:**

.. code-block:: python

   # Position sizing with daily volatility
   daily_vol = belief.get_vol(annualize=False, as_percentage=False)
   position_size = capital * (target_risk / daily_vol)
   
   # Risk-adjusted returns (Sharpe ratio)
   annual_vol = belief.get_vol(annualize=True, as_percentage=False)
   sharpe = annual_return / annual_vol
   
   # Simple monitoring (human-readable)
   vol_pct = belief.vol  # Default: annualized %
   print(f"Current vol: {vol_pct:.1f}%")

var_h: Uncertainty
^^^^^^^^^^^^^^^^^^

Posterior variance of log-volatility. Measures how confident the filter is.

**Range:** 0 to ~5 (typical: 0.01-0.50)

**Usage:**

- Model confidence indicator
- High var_h → widen stops, reduce leverage
- Low var_h → tighten stops, increase confidence

**Example:**

.. code-block:: python

   if features['var_h'] > 0.3:
       # High uncertainty - reduce position sizes
       confidence_multiplier = 0.5
   else:
       confidence_multiplier = 1.0

weight_entropy: Regime Ambiguity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Entropy of model weights (in bits). High entropy means multiple regimes are
plausible, low entropy means one regime dominates.

**Range:** 0 to log₂(K) bits (50 models → max 5.64 bits)

**Usage:**

- Regime transition detector
- High entropy → unclear regime, risky to trade on volatility
- Low entropy → clear regime, stronger signal

**Example:**

.. code-block:: python

   max_entropy = math.log2(config.num_models)
   normalized_entropy = features['weight_entropy'] / max_entropy
   
   if normalized_entropy > 0.8:
       # Very uncertain regime - avoid volatility-based trades
       skip_trade = True

weighted_phi: Persistence Regime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bayesian Model Average (BMA) of persistence parameter φ.

**Range:** [0.90, 0.99] (based on your parameter grid)

**Interpretation:**

- φ = 0.90 → volatility shocks decay quickly (half-life ~7 periods)
- φ = 0.95 → moderate persistence (half-life ~14 periods)
- φ = 0.99 → very persistent (half-life ~69 periods)

**Usage:**

- High φ → recent volatility spike will persist, stay defensive
- Low φ → volatility mean-reverts quickly, fade the move

**Example:**

.. code-block:: python

   if features['weighted_phi'] > 0.97:
       # High persistence regime - volatility shock lasts
       holding_period *= 2  # Extend time horizon
   else:
       # Low persistence - quick mean reversion
       holding_period *= 0.5

weighted_nu: Tail Risk Regime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BMA of Student-t degrees of freedom ν.

**Range:** >2 (typical: 3-20, based on your grid)

**Interpretation:**

- ν → 2: Extremely heavy tails, high kurtosis, frequent extremes
- ν = 5: Heavy tails, notable kurtosis
- ν = 10: Moderate tails
- ν > 20: Near-Gaussian

**Usage:**

- Low ν → expect extreme moves, widen stops
- High ν → normal distribution likely, standard risk management

**Example:**

.. code-block:: python

   if features['weighted_nu'] < 5:
       # Heavy-tail regime - prepare for large moves
       stop_loss_width *= 1.5
       tail_hedge = True
   else:
       stop_loss_width *= 1.0
       tail_hedge = False

log_likelihood: Model Fit
^^^^^^^^^^^^^^^^^^^^^^^^^^

One-step-ahead predictive log p(y_t | y_{1:t-1}).

**Range:** -∞ to ~10 (typical: -5 to +5)

**Usage:**

- Model quality indicator
- Sudden drop in log-likelihood → regime change, structural break
- Rising log-likelihood → model fitting well

**Example:**

.. code-block:: python

   # Track rolling average
   avg_loglik = np.mean([f['log_likelihood'] for f in features_history[-20:]])
   
   if features['log_likelihood'] < avg_loglik - 3:
       # Model fit degraded significantly
       regime_change = True
       reduce_positions = True

Feature Engineering Tips
------------------------

**Normalization:**

.. code-block:: python

   # Normalize entropy to [0, 1]
   norm_entropy = weight_entropy / math.log2(num_models)
   
   # Z-score volatility
   vol_zscore = (vol - vol_mean) / vol_std

**Lagged Features:**

.. code-block:: python

   # Changes in regime
   delta_phi = weighted_phi - weighted_phi_prev
   delta_entropy = weight_entropy - weight_entropy_prev

**Regime Indicators:**

.. code-block:: python

   # Binary flags
   high_persistence = int(weighted_phi > 0.97)
   tail_risk = int(weighted_nu < 6)
   regime_unclear = int(norm_entropy > 0.7)

**Interaction Terms:**

.. code-block:: python

   # Uncertainty-adjusted volatility
   adjusted_vol = vol * (1 + var_h)
   
   # Risk-weighted persistence
   risk_metric = weighted_phi * (1 / weighted_nu)

Storage Format
--------------

Efficient storage for backtesting or training:

.. code-block:: python

   import pandas as pd
   
   df = pd.DataFrame(features_history)
   df.index = timestamps  # Align with returns
   
   # Save to disk
   df.to_parquet('volatility_features.parquet')
   
   # Or numpy for speed
   np.savez_compressed('features.npz',
       vol=df['vol'].values,
       var_h=df['var_h'].values,
       entropy=df['weight_entropy'].values,
       phi=df['weighted_phi'].values,
       nu=df['weighted_nu'].values,
       loglik=df['log_likelihood'].values)

Performance Considerations
--------------------------

**Computational Cost:**

- ``step()``: O(K × N) - main filter update
- ``get_belief()``: O(1) - just returns cached state
- ``get_weighted_params()``: O(K) - compute weighted average
- ``get_weight_entropy()``: O(K) - sum over weights

All methods are fast (sub-millisecond) for typical K=50, N=500.

**Memory:**

Each ``Svmix`` instance uses ~40MB (50 models × 500 particles × 16 bytes).
Feature extraction adds negligible overhead.

API Reference
-------------

See :meth:`svmix.Svmix.get_weighted_params` and
:meth:`svmix.Svmix.get_weight_entropy` for detailed API documentation.
