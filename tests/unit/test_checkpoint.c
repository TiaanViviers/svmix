/**
 * @file test_checkpoint.c
 * @brief Comprehensive tests for checkpoint save/load functionality.
 */

#include "../test_common.h"
#include "../../include/svmix.h"
#include "../../src/checkpoint.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

/* Test helpers */
static svmix_cfg_t default_config(void) {
    svmix_cfg_t cfg;
    cfg.num_models = 3;
    cfg.num_particles = 500;
    cfg.spec = SVMIX_SPEC_VOL;
    cfg.ensemble.lambda = 0.99;
    cfg.ensemble.beta = 1.0;
    cfg.ensemble.epsilon = 0.05;
    cfg.ensemble.num_threads = 0;
    return cfg;
}

static svmix_sv_params_t default_params(void) {
    svmix_sv_params_t params;
    params.mu_h = -13.815510557964;
    params.phi_h = 0.98;
    params.sigma_h = 0.15;
    params.nu = 7.0;
    return params;
}

static void run_steps(svmix_t* sv, const double* observations, size_t n_steps) {
    for (size_t i = 0; i < n_steps; i++) {
        svmix_step(sv, observations[i]);
    }
}

static void generate_observations(double* obs, size_t n, unsigned long seed) {
    for (size_t i = 0; i < n; i++) {
        obs[i] = 0.01 * sin((double)i / 10.0 + seed);
    }
}

/* Test 1: Basic roundtrip */
TEST(checkpoint_basic_roundtrip) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {1000, 1001, 1002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    svmix_t* sv = svmix_create(&cfg, params, seeds);
    ASSERT_TRUE(sv != NULL);
    
    double obs[50];
    generate_observations(obs, 50, 1000);
    run_steps(sv, obs, 50);
    
    const char* filename = "/tmp/svmix_test.svmix";
    svmix_status_t status = svmix_save_checkpoint(sv, filename);
    ASSERT_EQ(status, SVMIX_OK);
    
    svmix_status_t load_status;
    svmix_t* sv_loaded = svmix_load_checkpoint(filename, &load_status);
    ASSERT_EQ(load_status, SVMIX_OK);
    ASSERT_TRUE(sv_loaded != NULL);
    ASSERT_EQ(svmix_get_timestep(sv_loaded), 50);
    
    svmix_free(sv);
    svmix_free(sv_loaded);
    remove(filename);
}

/* Test 2: Deterministic resume - CRITICAL! */
TEST(checkpoint_deterministic_resume) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {2000, 2001, 2002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    double obs[100];
    generate_observations(obs, 100, 2000);
    
    /* Path A: run 50, save, run 50 more */
    svmix_t* sv_a = svmix_create(&cfg, params, seeds);
    run_steps(sv_a, obs, 50);
    
    const char* filename = "/tmp/svmix_determ.svmix";
    svmix_save_checkpoint(sv_a, filename);
    run_steps(sv_a, obs + 50, 50);
    
    /* Path B: load, run 50 */
    svmix_status_t load_status;
    svmix_t* sv_b = svmix_load_checkpoint(filename, &load_status);
    ASSERT_EQ(load_status, SVMIX_OK);
    run_steps(sv_b, obs + 50, 50);
    
    /* Compare - must be bit-exact */
    double weights_a[3], weights_b[3];
    svmix_get_weights(sv_a, weights_a, 3);
    svmix_get_weights(sv_b, weights_b, 3);
    
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ_DOUBLE(weights_a[i], weights_b[i]);
    }
    
    svmix_free(sv_a);
    svmix_free(sv_b);
    remove(filename);
}

/* Test 3: Single model */
TEST(checkpoint_single_model) {
    svmix_cfg_t cfg = default_config();
    cfg.num_models = 1;
    
    svmix_sv_params_t params[1];
    params[0] = default_params();
    unsigned long seeds[1] = {3000};
    
    svmix_t* sv = svmix_create(&cfg, params, seeds);
    double obs[20];
    generate_observations(obs, 20, 3000);
    run_steps(sv, obs, 20);
    
    const char* filename = "/tmp/svmix_single.svmix";
    svmix_save_checkpoint(sv, filename);
    
    svmix_status_t load_status;
    svmix_t* sv_loaded = svmix_load_checkpoint(filename, &load_status);
    ASSERT_EQ(load_status, SVMIX_OK);
    ASSERT_EQ(svmix_get_num_models(sv_loaded), 1);
    
    svmix_free(sv);
    svmix_free(sv_loaded);
    remove(filename);
}

/* Test 4: Corrupted magic */
TEST(checkpoint_corrupted_magic) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {4000, 4001, 4002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    svmix_t* sv = svmix_create(&cfg, params, seeds);
    
    const char* filename = "/tmp/svmix_corrupt.svmix";
    svmix_save_checkpoint(sv, filename);
    
    /* Corrupt magic */
    FILE* fp = fopen(filename, "r+b");
    fseek(fp, 0, SEEK_SET);
    const char bad_magic[8] = "BADMAGIC";
    fwrite(bad_magic, 1, 8, fp);
    fclose(fp);
    
    svmix_status_t load_status;
    svmix_t* sv_loaded = svmix_load_checkpoint(filename, &load_status);
    ASSERT_TRUE(sv_loaded == NULL);
    ASSERT_EQ(load_status, SVMIX_ERR_CHECKPOINT_CORRUPT);
    
    svmix_free(sv);
    remove(filename);
}

/* Test 5: Version mismatch */
TEST(checkpoint_version_mismatch) {
    svmix_cfg_t cfg = default_config();
    svmix_sv_params_t params[3];
    unsigned long seeds[3] = {5000, 5001, 5002};
    
    for (int i = 0; i < 3; i++) {
        params[i] = default_params();
    }
    
    svmix_t* sv = svmix_create(&cfg, params, seeds);
    
    const char* filename = "/tmp/svmix_version.svmix";
    svmix_save_checkpoint(sv, filename);
    
    /* Corrupt version */
    FILE* fp = fopen(filename, "r+b");
    fseek(fp, 8, SEEK_SET);
    uint32_t bad_version = 99;
    fwrite(&bad_version, sizeof(uint32_t), 1, fp);
    fclose(fp);
    
    svmix_status_t load_status;
    svmix_t* sv_loaded = svmix_load_checkpoint(filename, &load_status);
    ASSERT_TRUE(sv_loaded == NULL);
    ASSERT_EQ(load_status, SVMIX_ERR_VERSION_MISMATCH);
    
    svmix_free(sv);
    remove(filename);
}

/* Test 6: NULL inputs */
TEST(checkpoint_null_inputs) {
    /* Save with NULL svmix */
    svmix_status_t status = svmix_save_checkpoint(NULL, "/tmp/test.svmix");
    ASSERT_EQ(status, SVMIX_ERR_NULL_POINTER);
    
    /* Load with NULL filename */
    svmix_status_t load_status;
    svmix_t* sv = svmix_load_checkpoint(NULL, &load_status);
    ASSERT_TRUE(sv == NULL);
    ASSERT_EQ(load_status, SVMIX_ERR_NULL_POINTER);
}

int main(void) {
    printf("========================================\n");
    printf("svmix Checkpoint Tests\n");
    printf("========================================\n\n");
    
    RUN_TEST(checkpoint_basic_roundtrip);
    RUN_TEST(checkpoint_deterministic_resume);
    RUN_TEST(checkpoint_single_model);
    RUN_TEST(checkpoint_corrupted_magic);
    RUN_TEST(checkpoint_version_mismatch);
    RUN_TEST(checkpoint_null_inputs);
    
    return test_summary();
}
