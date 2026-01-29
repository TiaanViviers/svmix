"""
Common utilities for benchmarking svmix performance.
"""
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path to import svmix
sys.path.insert(0, '../../python')
from svmix import Svmix, SvmixConfig, SvParams, Spec


def config_model(model_count, particle_count, thread_count, seed=42, spec="vol"):
    """
    Configure a Svmix model with standard parameters.
    
    Parameters:
    -----------
    model_count : int
        Number of models in ensemble (K)
    particle_count : int
        Number of particles per model (N)
    thread_count : int
        Number of OpenMP threads (0=auto)
    seed : int
        Random seed for reproducibility
    spec : str
        Model specification ('vol', 'drift', 'vol_drift')
    
    Returns:
    --------
    Svmix
        Configured model instance
    """
    if spec == "vol":
        spec_enum = Spec.VOL
    else:
        raise ValueError(f"Unknown spec: {spec}")

    config = SvmixConfig(
        spec=spec_enum,
        num_models=model_count,
        num_particles=particle_count,
        lambda_=0.995,
        epsilon=1e-6,
        beta=1.0,
        num_threads=thread_count,
        seed=seed
    )
    
    # Generate parameter grid spanning phi range
    params = SvParams.linspace(
        num_models=model_count,
        phi=(0.93, 0.99),
        sigma=0.20,
        nu=10.0,
        mu=-0.5
    )

    return Svmix(config, params)


def speed_test(model, observations):
    """
    Time how long it takes to process observations.
    
    Parameters:
    -----------
    model : Svmix
        The model to benchmark
    observations : np.ndarray
        Time series data
    
    Returns:
    --------
    float
        Elapsed time in seconds
    """
    start_t = time.perf_counter()
    
    for obs in observations:
        model.step(obs)
    
    elapsed = time.perf_counter() - start_t
    
    return elapsed


def read_data(filename="synthetic.npy"):
    """
    Read synthetic benchmark data.
    
    Parameters:
    -----------
    filename : str
        Name of data file in perf/data/
    
    Returns:
    --------
    np.ndarray
        Time series observations
    """
    data_dir = Path(__file__).parent.parent / "data"
    filepath = data_dir / filename
    return np.load(filepath)



