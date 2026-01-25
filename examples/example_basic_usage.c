/**
 * @file example_basic_usage.c
 * @brief Basic usage example for svmix API.
 *
 * Demonstrates:
 * - Creating an ensemble with 3 models
 * - Stepping through synthetic return data
 * - Retrieving belief and weights
 * - Proper cleanup
 */

#include "svmix.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("===========================================\n");
    printf("svmix Basic Usage Example\n");
    printf("Version: %s\n", svmix_version());
    printf("===========================================\n\n");
    
    /* ====================================================================
     * Step 1: Configure ensemble
     * ==================================================================== */
    
    svmix_cfg_t cfg = {
        .num_models = 3,
        .num_particles = 1000,
        .spec = SVMIX_SPEC_VOL,
        .ensemble = {
            .lambda = 0.99,       /* Exponential forgetting */
            .beta = 1.0,          /* Softmax temperature */
            .epsilon = 0.05,      /* Anti-starvation (5% floor) */
            .num_threads = 0      /* Auto thread count */
        }
    };
    
    /* ====================================================================
     * Step 2: Define model parameters (K=3 models with different phi)
     * ==================================================================== */
    
    svmix_sv_params_t models[3] = {
        {.mu_h = -13.82, .phi_h = 0.96, .sigma_h = 0.15, .nu = 7.0},  /* Low persistence */
        {.mu_h = -13.82, .phi_h = 0.98, .sigma_h = 0.15, .nu = 7.0},  /* Medium persistence */
        {.mu_h = -13.82, .phi_h = 0.99, .sigma_h = 0.15, .nu = 7.0}   /* High persistence */
    };
    
    unsigned long seeds[3] = {42, 43, 44};
    
    /* ====================================================================
     * Step 3: Create ensemble
     * ==================================================================== */
    
    printf("Creating ensemble with K=%zu models, N=%zu particles each...\n",
           cfg.num_models, cfg.num_particles);
    
    svmix_t* sv = svmix_create(&cfg, models, seeds);
    if (!sv) {
        fprintf(stderr, "ERROR: Failed to create ensemble\n");
        return 1;
    }
    
    printf("Ensemble created successfully\n\n");
    
    /* ====================================================================
     * Step 4: Get initial belief
     * ==================================================================== */
    
    svmix_belief_t belief;
    if (svmix_get_belief(sv, &belief) != SVMIX_OK) {
        fprintf(stderr, "ERROR: Failed to get belief\n");
        svmix_free(sv);
        return 1;
    }
    
    printf("Initial state:\n");
    printf("  mean_h:      %12.6f (log-variance)\n", belief.mean_h);
    printf("  var_h:       %12.6f\n", belief.var_h);
    printf("  mean_sigma:  %12.8f (approx volatility)\n", belief.mean_sigma);
    printf("  valid:       %s\n\n", belief.valid ? "yes" : "no");
    
    double weights[3];
    if (svmix_get_weights(sv, weights, 3) != SVMIX_OK) {
        fprintf(stderr, "ERROR: Failed to get weights\n");
        svmix_free(sv);
        return 1;
    }
    
    printf("Initial weights (uniform):\n");
    for (size_t i = 0; i < 3; i++) {
        printf("  Model %zu (phi=%.2f): %.4f\n", i, models[i].phi_h, weights[i]);
    }
    printf("\n");
    
    /* ====================================================================
     * Step 5: Process synthetic return data (100 observations)
     * ==================================================================== */
    
    printf("Processing 100 observations...\n");
    
    for (int t = 0; t < 100; t++) {
        /* Synthetic returns: small oscillating pattern */
        double r_t = 0.002 * sin((double)t * 0.1);
        
        svmix_status_t status = svmix_step(sv, r_t);
        if (status != SVMIX_OK) {
            fprintf(stderr, "ERROR at t=%d: %s\n", t, svmix_status_string(status));
            svmix_free(sv);
            return 1;
        }
        
        /* Print progress every 20 steps */
        if ((t + 1) % 20 == 0) {
            if (svmix_get_belief(sv, &belief) == SVMIX_OK) {
                printf("  t=%3d: sigma=%.8f, mean_h=%.4f\n",
                       t + 1, belief.mean_sigma, belief.mean_h);
            }
        }
    }
    
    printf("\nProcessed %zu timesteps\n\n", svmix_get_timestep(sv));
    
    /* ====================================================================
     * Step 6: Get final state
     * ==================================================================== */
    
    if (svmix_get_belief(sv, &belief) != SVMIX_OK) {
        fprintf(stderr, "ERROR: Failed to get final belief\n");
        svmix_free(sv);
        return 1;
    }
    
    printf("Final state:\n");
    printf("  mean_h:      %12.6f\n", belief.mean_h);
    printf("  var_h:       %12.6f\n", belief.var_h);
    printf("  mean_sigma:  %12.8f\n", belief.mean_sigma);
    printf("  valid:       %s\n\n", belief.valid ? "yes" : "no");
    
    if (svmix_get_weights(sv, weights, 3) != SVMIX_OK) {
        fprintf(stderr, "ERROR: Failed to get final weights\n");
        svmix_free(sv);
        return 1;
    }
    
    printf("Final weights (adapted):\n");
    for (size_t i = 0; i < 3; i++) {
        printf("  Model %zu (phi=%.2f): %.4f\n", i, models[i].phi_h, weights[i]);
    }
    printf("\n");
    
    /* ====================================================================
     * Step 7: Cleanup
     * ==================================================================== */
    
    svmix_free(sv);
    printf("Ensemble freed\n");
    printf("\nSuccess!\n");
    
    return 0;
}
