Data Requirements
=================

This page describes the input data format and preprocessing requirements
for svmix.

Return Transformation
---------------------

**CRITICAL**: svmix expects **log returns**, not simple/percentage returns.

The stochastic volatility model is mathematically specified for log returns:

.. math::

   r_t = \log(P_t / P_{t-1}) = \log(P_t) - \log(P_{t-1})

where :math:`P_t` is the price at time :math:`t`.

Do NOT use simple returns:

.. math::

   R_t = \frac{P_t - P_{t-1}}{P_{t-1}}  \quad \text{(INCORRECT)}

Why Log Returns?
^^^^^^^^^^^^^^^^

**Mathematical Requirements:**

1. **Time Additivity**: Multi-period log returns are sums of single-period
   log returns:

   .. math::

      \log(P_T / P_0) = \sum_{t=1}^T \log(P_t / P_{t-1})

   This property does NOT hold for simple returns, breaking the model's
   time-series structure.

2. **Symmetry**: Log returns treat gains and losses symmetrically, which
   aligns with the symmetric Student-t observation model.

3. **Variance Interpretation**: The log-volatility process :math:`h_t`
   directly models the variance of log returns.

**Statistical Properties:**

- Log returns exhibit approximate normality (before adding stochastic
  volatility), which is the baseline assumption of the model.

- Simple returns have an asymmetric distribution that violates model
  assumptions.

- For small returns (< 2%), log returns :math:`\approx` simple returns,
  but the difference compounds over time.

**Research Consensus:**

All stochastic volatility literature uses log returns:

- Kim, Shephard & Chib (1998)
- Jacquier, Polson & Rossi (2004)
- Shephard & Pitt (1997)

Using simple returns introduces bias in volatility estimates that
increases with return magnitude.

Practical Impact
^^^^^^^^^^^^^^^^

The impact of using the wrong return type depends on the data:

.. list-table::
   :header-rows: 1

   * - Asset Class
     - Typical Move Size
     - Impact of Wrong Transform
   * - Daily equity
     - 1-2%
     - Negligible (< 0.02%)
   * - Daily crypto
     - 3-10%
     - Moderate (0.1-0.5%)
   * - Crisis periods
     - 5-20%
     - **Severe** (1-5%)
   * - Intraday (1-min)
     - 0.1-0.5%
     - Minimal

**Bottom line**: Always use log returns. The computational cost is
negligible, and it ensures model validity.

Correct Implementation
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import pandas as pd

   # Load price data
   df = pd.read_csv('prices.csv')

   # Compute log returns (CORRECT)
   returns = np.log(df['close'] / df['close'].shift(1))

   # Drop the first NaN value
   returns = returns.dropna()

   # Pass to svmix
   from svmix import Svmix, SvmixConfig, SvParams, Spec

   config = SvmixConfig(
       spec=Spec.VOL,
       num_models=50,
       num_particles=500,
       lambda_=0.995,
       beta=0.8,
       epsilon=0.05
   )

   params = SvParams.linspace(
       num_models=50,
       phi=(0.90, 0.99),
       sigma=0.2,
       nu=10,
       mu=-0.5
   )

   svmix = Svmix(config, params)

   for r in returns:
       svmix.step(r)

   svmix.free()

Common Mistakes
^^^^^^^^^^^^^^^

**WRONG**: Using pandas ``pct_change()``

.. code-block:: python

   # This produces simple returns, not log returns
   returns = df['close'].pct_change()  # WRONG!

**WRONG**: Using price differences

.. code-block:: python

   # This is not even normalized
   returns = df['close'].diff()  # WRONG!

**CORRECT**: Using numpy log and division

.. code-block:: python

   # This produces log returns
   returns = np.log(df['close'] / df['close'].shift(1))  # CORRECT!

   # Equivalent alternative
   returns = np.log(df['close']).diff()  # Also CORRECT!

Data Frequency
--------------

svmix is designed for regularly-spaced time-series data. Supported
frequencies:

- **High-frequency**: 1-minute, 5-minute bars (requires fast execution)
- **Daily**: Most common for equity/crypto volatility estimation
- **Weekly**: For long-term trend analysis

The model does NOT handle:

- Irregular time spacing (use interpolation first)
- Missing data (forward-fill or drop gaps)
- Multiple observations per timestep (aggregate first)

Data Cleaning
-------------

Before passing data to svmix:

1. **Remove outliers**: Log returns > 50% (±0.4 in log space) indicate
   data errors or corporate actions (splits, dividends).

2. **Handle gaps**: Forward-fill short gaps (< 3 periods) or restart
   the filter after long gaps.

3. **Check for stationarity**: If mean return is significantly non-zero,
   consider demeaning or modeling drift separately.

4. **Verify scale**: Daily log returns typically range from -0.05 to
   +0.05 (±5%). Values outside -0.2 to +0.2 warrant investigation.

Example preprocessing pipeline:

.. code-block:: python

   import numpy as np
   import pandas as pd

   def preprocess_prices(df, price_col='close'):
       """Clean price data and compute log returns."""
       # Sort by date
       df = df.sort_values('date').reset_index(drop=True)

       # Remove duplicates
       df = df.drop_duplicates(subset='date')

       # Forward-fill missing prices (up to 3 days)
       df[price_col] = df[price_col].ffill(limit=3)

       # Drop remaining NaN
       df = df.dropna(subset=[price_col])

       # Compute log returns
       df['returns'] = np.log(df[price_col] / df[price_col].shift(1))

       # Remove outliers (>40% absolute log return)
       mask = df['returns'].abs() < 0.4
       df = df[mask]

       # Drop initial NaN
       df = df.dropna(subset=['returns'])

       return df

   # Usage
   df = pd.read_csv('prices.csv')
   df = preprocess_prices(df)
   returns = df['returns'].values

Initial Conditions
------------------

The first few observations are used to initialize the particle filter
state. Expect:

- Higher uncertainty in the first 10-20 steps
- Gradual convergence as the filter ingests data
- Better initial estimates with warmer starting parameters

For production systems, consider:

- Warm-starting from historical estimates
- Loading from checkpoint after initial burn-in
- Discarding the first N estimates as unreliable
