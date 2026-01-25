# OpenMP Support in svmix

## Overview

svmix supports **optional** OpenMP parallelization for ensemble model stepping. When enabled, the K models in an ensemble step through observations in parallel, significantly reducing wall-clock time for large ensembles.

## Current Implementation

### What's Parallelized

The core parallel region is in `ensemble_step()`:

```c
#ifdef _OPENMP
    #pragma omp parallel for
#endif
for (size_t i = 0; i < ens->K; i++) {
    // Step model i's particle filter
    fastpf_step(model->pf, &observation);
    // Update score: S_i = lambda * S_i + log_norm_const
    model->score = lambda * model->score + diag->log_norm_const;
}
```

**After** parallel stepping, we serially compute weights (this is cheap and avoids race conditions):

```c
// Serial: collect scores and compute weights
for (size_t i = 0; i < K; i++) {
    scores[i] = models[i].score;
}
compute_weights(scores, K, beta, epsilon, weights);  // Log-stable softmax
```

### What's NOT Parallelized

- **Per-model particle filters**: fastpf v1 doesn't expose internal OpenMP, so each `fastpf_step()` runs serially
- **Weight computation**: Cheap O(K) operation, not worth parallelizing
- **Belief aggregation**: Also O(K), runs serially

### Performance Characteristics

- **Best case**: K large (e.g., K=50 models), each model has many particles (N=5000+)
  - Wall-clock time ≈ (single model time) / num_threads
  
- **Diminishing returns**: K small (e.g., K=5 models)
  - Overhead may outweigh benefits if K < num_threads
  
- **No benefit**: K=1 (single model)
  - Use serial build (no `-fopenmp`)

## Building with OpenMP

### Default (Serial)

```bash
make test              # All 47 tests, OpenMP tests gracefully skip
make clean && make all # Serial build
```

**Output for OpenMP tests:**
```
========================================
Ensemble OpenMP Tests
========================================

  Running test: openmp_enabled ... 
    INFO: OpenMP not enabled (compile with OPENMP=1 to enable)
    Skipping OpenMP tests - this is expected behavior
PASS
  Running test: thread_count_explicit ... 
    SKIP: OpenMP not enabled
PASS
  Running test: thread_count_auto ... 
    SKIP: OpenMP not enabled
PASS
  Running test: determinism_with_openmp ... 
    SKIP: OpenMP not enabled
PASS

========================================
All tests passed! (4/4)
========================================
```

All tests **pass** (they skip gracefully), so CI/CD won't break on systems without OpenMP.

### With OpenMP

```bash
make OPENMP=1 test                    # All 47 tests with OpenMP enabled
make OPENMP=1 test-ensemble-openmp    # Only OpenMP-specific tests
make clean && make OPENMP=1 all       # Full parallel build
```

**Note:** `OPENMP=1` means "OpenMP is **enabled**" (like a boolean flag), NOT "use 1 thread".

The `OPENMP=1` flag adds `-fopenmp` to both `CFLAGS` and `LDFLAGS`.

**Output for OpenMP tests:**
```
========================================
Ensemble OpenMP Tests
========================================

  Running test: openmp_enabled ... 
    OpenMP version: 201511
    Max threads available: 16
PASS
  Running test: thread_count_explicit ... 
    Requested: 2 threads, Actual: 2 threads
PASS
  Running test: thread_count_auto ... 
    Auto thread count: 16
PASS
  Running test: determinism_with_openmp ... 
    Determinism verified with 2 OpenMP threads
PASS

========================================
All tests passed! (4/4)
========================================
```

## Thread Control

### Via Ensemble Configuration

```c
ensemble_config_t config = {
    .lambda = 0.99,
    .beta = 1.0,
    .epsilon = 0.05,
    .num_threads = 4  // <-- Explicit thread count
};
```

**Behavior:**
- `num_threads = 0`: Auto (uses `$OMP_NUM_THREADS` or system default)
- `num_threads > 0`: Explicitly set thread count (e.g., `4` = use 4 threads)

Example:
```c
// Use 8 threads regardless of OMP_NUM_THREADS
config.num_threads = 8;
ensemble_t* ens = ensemble_create(K, params, 1000, seeds, &config);
```

### Via Environment Variable

If `num_threads = 0`, OpenMP respects `$OMP_NUM_THREADS`:

```bash
export OMP_NUM_THREADS=4
./my_program  # Will use 4 threads
```

## Determinism Guarantees

**OpenMP parallelization does NOT break determinism** as long as:

1. **Same RNG seeds** for each model
2. **Same thread count** (important! 2 threads ≠ 4 threads in floating-point order)
3. **Same input stream**

Our tests verify this:

```c
TEST(determinism_with_openmp) {
    // Run 1 with 2 threads
    config.num_threads = 2;
    ensemble_t* ens1 = ensemble_create(K, params, N, seeds, &config);
    for (int t = 0; t < 30; t++) {
        ensemble_step(ens1, observations[t]);
    }
    double weights1[K];
    ensemble_get_weights(ens1, weights1);
    
    // Run 2 with same seeds and 2 threads
    ensemble_t* ens2 = ensemble_create(K, params, N, seeds, &config);
    for (int t = 0; t < 30; t++) {
        ensemble_step(ens2, observations[t]);
    }
    double weights2[K];
    ensemble_get_weights(ens2, weights2);
    
    // Bitwise-identical results
    for (size_t i = 0; i < K; i++) {
        assert(weights1[i] == weights2[i]);  // Within 1e-15
    }
}
```

## Testing

### Run All Tests (Including OpenMP)

```bash
# Without OpenMP (serial, OpenMP tests skip gracefully)
make test
# Output: All 47 tests pass (11+13+9+4+6+4)

# With OpenMP (parallel, OpenMP tests actually run)
make OPENMP=1 test
# Output: All 47 tests pass, OpenMP tests verify parallelization
```

### Verify OpenMP is Enabled

```bash
make OPENMP=1 test-ensemble-openmp
```

Expected output:
```
  Running test: openmp_enabled ... 
    OpenMP version: 201511
    Max threads available: 16
PASS
  Running test: thread_count_explicit ... 
    Requested: 2 threads, Actual: 2 threads
PASS
```

### Without OpenMP

```bash
make test  # All tests run, OpenMP tests skip
```

Expected output:
```
Ensemble OpenMP Tests
========================================

  Running test: openmp_enabled ... 
    INFO: OpenMP not enabled (compile with OPENMP=1 to enable)
    Skipping OpenMP tests - this is expected behavior
PASS
```

All 4 OpenMP tests **pass** (they skip gracefully) - this is **by design** so CI/CD doesn't break.

## Compilation Details

### Without OPENMP=1

```bash
gcc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
    -Iinclude -Isrc -Ithird_party/fastpf/include \
    src/ensemble.c -c -o bin/ensemble.o
```

Result:
- `_OPENMP` macro **not defined**
- `#pragma omp parallel for` is **ignored** (treated as comment)
- No OpenMP runtime linked
- Code runs serially
- Still compiles and runs correctly!

### With OPENMP=1

```bash
gcc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -fopenmp \
    -Iinclude -Isrc -Ithird_party/fastpf/include \
    src/ensemble.c -c -o bin/ensemble.o
```

Result:
- `_OPENMP` macro **defined** (value = OpenMP version, e.g., 201511)
- `#pragma omp parallel for` is **active**
- OpenMP runtime linked (`-fopenmp` in LDFLAGS)
- Code runs in parallel

## Future Considerations

### fastpf Internal Parallelism

Currently, each model's `fastpf_step()` runs serially. If fastpf adds internal OpenMP (e.g., parallel particle propagation), we should:

1. **Nested parallelism**: Set `OMP_NESTED=true` or use `omp_set_nested(1)`
2. **Thread allocation**: Divide threads between ensemble-level (K models) and PF-level (N particles)
   - Example: 16 threads total, K=4 models → 4 threads per model

### Model-Level Load Balancing

If models have vastly different costs (e.g., some resample, others don't), consider:

```c
#pragma omp parallel for schedule(dynamic)
```

This lets OpenMP assign models to threads dynamically instead of static round-robin.

### Python Bindings

When exposing to Python, users will configure threads via:

```python
config = svmix.EnsembleConfig(
    lambda_=0.99,
    beta=1.0,
    epsilon=0.05,
    num_threads=4  # <-- Pythonic API
)
ens = svmix.Ensemble(models=..., config=config)
```

Internally, this maps directly to the C `ensemble_config_t.num_threads`.

## Recommended Settings

| Use Case | K (models) | N (particles) | num_threads | Rationale |
|----------|-----------|---------------|-------------|-----------|
| **Prototyping** | 5-10 | 1000 | 0 (auto) | Let OpenMP decide |
| **Production** | 20-50 | 5000+ | 8-16 | Explicit control, high throughput |
| **Debugging** | 1-5 | 500 | 0 or 1 | Serial easier to debug |
| **Embedded** | 2-5 | 1000 | 2-4 | Limited cores |

## Summary

✅ **OpenMP is optional** - code compiles and runs without `-fopenmp`  
✅ **Determinism preserved** - same seeds + same threads = bitwise-identical results  
✅ **User-controlled** - explicit thread count via `ensemble_config_t.num_threads`  
✅ **Future-proof** - ready for fastpf internal parallelism when available  
✅ **Well-tested** - 4 OpenMP-specific tests verify thread control and determinism  

**Bottom line:** For production with K≥10 and adequate CPU cores, use `OPENMP=1` and set `num_threads` explicitly for reproducibility.
