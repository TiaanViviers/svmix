import sys
import numpy as np
import time
from pathlib import Path

# Add parent directory to path to import svmix
sys.path.insert(0, '../../python')
from svmix import Svmix, SvmixConfig, SvParams, Spec

MODEL_COUNTS = [10, 50, 100]
PARTICLE_COUNTS = [1024, 4096, 10_000]
THREAD_COUNTS = [0, 4, 8, 16]
SPEC = "vol"


def main():
    """
    Main entry point for the thread benchmarking.
    """
    print("=" * 60)
    print("Running thread benchmark with:")
    print(f"  Model counts: {MODEL_COUNTS}")
    print(f"  Particle counts: {PARTICLE_COUNTS}")
    print(f"  Thread counts: {THREAD_COUNTS}")

    observations = read_data()
    print(f"  Number of observations: {len(observations)}")
    print("=" * 60)
    print("\n")
    
    # Collect results for summary
    results = []
    
    for model_count in MODEL_COUNTS:
        for particle_count in PARTICLE_COUNTS:
            config_results = {}
            for thread_count in THREAD_COUNTS:
                model = config_model(model_count, particle_count, thread_count)
                elapsed_time = speed_test(model, observations)
                config_results[thread_count] = elapsed_time
                print_results(model_count, particle_count, thread_count, elapsed_time, len(observations))
                model.free()
            
            results.append({
                'K': model_count,
                'N': particle_count,
                'times': config_results
            })
            print()

    print("=" * 60)
    print_summary(results, len(observations))
    print("=" * 60)


def speed_test(model, observations):
    """Run speed tests on the model using the provided observations.

    Parameters:
    -----------
    model : Svmix
        The Svmix model to test.
    observations : np.ndarray
        The synthetic observations to use for testing.
    """
    start_t = time.perf_counter()
    
    for obs in observations:
        model.step(obs)
    elapsed = time.perf_counter() - start_t
    
    return elapsed


def config_model(model_count, particle_count, thread_count):
    """Configure the Svmix model.
    
    Parameters:
    -----------
    model_count : int
        The number of models to use.
    particle_count : int
        The number of particles to use.
    thread_count : int
        The number of threads to use.
    """
    if SPEC == "vol":
        spec = Spec.VOL
    else:
        raise ValueError(f"Unknown spec: {SPEC}")

    config = SvmixConfig(
        spec=spec,
        num_models=model_count,
        num_particles=particle_count,
        lambda_=0.995,                  # Discount factor for model weights
        epsilon=1e-6,                   # Weight floor (anti-starvation)
        beta=1.0,                       # Tempering parameter
        num_threads=thread_count,
        seed=42
    )
    
    # Generate parameter grid
    params = SvParams.linspace(
        num_models=model_count,
        phi=(0.93, 0.99),
        sigma=(0.1, 0.3),
        nu=10.0,
        mu=-0.5
    )

    return Svmix(config, params)


def print_results(model_count, particle_count, thread_count, elapsed_time, num_obs):
    """Print the results of the speed test."""
    _str = f"Models: {model_count}, Particles: {particle_count}, \
    Threads: {thread_count}, \
    Total: {elapsed_time:.4f} s \
    Per step: ~{elapsed_time / num_obs:.4f} s"
    print(_str)


def print_summary(results, num_obs):
    """Print benchmark summary statistics."""
    print("Summary:")
    print()
    
    # Calculate speedups relative to baseline (threads=0)
    print("  Speedup vs baseline (threads=0):")
    print(f"  {'Configuration':<25} {'threads=4':<12} {'threads=8':<12} {'threads=16':<12}")
    print("  " + "-" * 61)
    
    best_configs = []
    for r in results:
        K, N = r['K'], r['N']
        times = r['times']
        baseline = times[0]
        
        speedups = {t: baseline / times[t] for t in [4, 8, 16]}
        best_thread = max(speedups.items(), key=lambda x: x[1])
        best_configs.append((K, N, best_thread[0], best_thread[1]))
        
        config_str = f"K={K:3d}, N={N:5d}"
        print(f"  {config_str:<25} {speedups[4]:>5.2f}x       {speedups[8]:>5.2f}x       {speedups[16]:>5.2f}x")
    
    # Aggregate statistics
    print()
    thread_wins = {}
    for _, _, thread, _ in best_configs:
        thread_wins[thread] = thread_wins.get(thread, 0) + 1
    
    best_overall = max(thread_wins.items(), key=lambda x: x[1])
    avg_speedup = np.mean([s for _, _, _, s in best_configs])
    max_speedup = max(s for _, _, _, s in best_configs)
    min_speedup = min(s for _, _, _, s in best_configs)
    
    print(f"  Best thread count:     {best_overall[0]} threads ({best_overall[1]}/{len(best_configs)} configurations)")
    
    # Throughput for best configurations
    print()
    print("  Best throughput per configuration:")
    print(f"  {'Configuration':<25} {'Time/obs':<12} {'Throughput':<15}")
    print("  " + "-" * 52)
    for r in results:
        K, N = r['K'], r['N']
        best_time = min(r['times'].values())
        time_per_obs = best_time / num_obs
        throughput = num_obs / best_time
        
        config_str = f"K={K:3d}, N={N:5d}"
        print(f"  {config_str:<25} {time_per_obs*1000:>6.2f} ms     {throughput:>8.1f} obs/s")


def read_data():
    """Read synthetic input data."""
    data_dir = Path(__file__).parent.parent / "data"
    filepath = data_dir / "synthetic.npy"
    return np.load(filepath)

if __name__ == "__main__":
    main()