import argparse
import numpy as np
from scipy import stats
from pathlib import Path

def main():
    """
    Main function to generate synthetic data.
    """
    args = get_args()
    data = generate_synthetic_data(
        phi=args.phi,
        sigma=args.sigma,
        nu=args.nu,
        mu=args.mu,
        T=args.T,
    )
    write_data(data)


def generate_synthetic_data(phi, sigma, nu, mu, T):
    """
    Generate synthetic log-volatility data using a stochastic volatility model.

    Parameters:
    -----------
    phi : float
        Persistence of volatility
    sigma : float
        Volatility of volatility
    nu : float
        Degrees of freedom
    mu : float
        Long-run mean of log-volatility
    T : int
        Number of time steps
    """
    np.random.seed(42)

    # Simulate the log-volatility process
    h = np.zeros(T)
    h[0] = mu
    for t in range(1, T):
        h[t] = mu + phi * (h[t-1] - mu) + sigma * np.random.standard_t(nu)
        
    # Convert to actual volatility (standard deviation)
    volatility = np.exp(h / 2.0)

    # Generate observations from the true volatility
    observations = np.zeros(T)
    for t in range(T):
        # Sample standardized Student-t shock
        eta_t = stats.t.rvs(df=nu)
        # Scale by true volatility
        observations[t] = volatility[t] * eta_t

    return observations


def write_data(data):
    """
    Save generated data to .npy file.
    
    Parameters:
    -----------
    data : np.ndarray
        Generated time series
    """
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    filename = f"synthetic.npy"
    filepath = data_dir / filename
    
    np.save(filepath, data)
    print(f"Generated {len(data):,} observations")
    print(f"Saved to: {filepath}")


def get_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi", type=float, default=0.97, help="Persistence of volatility")
    parser.add_argument("--sigma", type=float, default=0.20, help="Volatility of volatility")
    parser.add_argument("--nu", type=float, default=10.0, help="Degrees of freedom")
    parser.add_argument("--mu", type=float, default=-0.5, help="Long-run mean of log-volatility")
    parser.add_argument("--T", type=int, default=500, help="Number of time steps")
    return parser.parse_args()


if __name__ == "__main__":
    main()