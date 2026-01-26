/**
 * @file checkpoint.c
 * @brief Checkpoint save/load implementation for svmix.
 *
 * Format overview:
 *   [Header: 64 bytes]
 *   [Ensemble metadata: config + weights + scores + timestep]
 *   [Model 0: params + fastpf blob]
 *   [Model 1: params + fastpf blob]
 *   ...
 *   [Model K-1: params + fastpf blob]
 *
 * Design:
 *   - Spec-aware: param size depends on spec (VOL=32 bytes, future: VOL_DRIFT=56, etc.)
 *   - Self-describing: sizes embedded in fastpf blobs
 *   - Binary: no parsing overhead
 *   - Validated: magic, version, size checks
 *   - Direct struct access: uses internal headers (industry standard)
 */

#include "checkpoint.h"
#include "svmix_internal.h"
#include "ensemble_internal.h"
#include "model_sv_internal.h"
#include "../third_party/fastpf/include/fastpf.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ========================================================================
 * Helper: Get parameter block size for spec
 * ======================================================================== */

static size_t get_param_size(svmix_spec_t spec) {
    switch (spec) {
        case SVMIX_SPEC_VOL:
            return sizeof(sv_params_t);  /* 4 doubles = 32 bytes */
        /* V2: Add more specs here
        case SVMIX_SPEC_DRIFT:
            return sizeof(sv_drift_params_t);
        case SVMIX_SPEC_VOL_DRIFT:
            return sizeof(sv_vol_drift_params_t);
        */
        default:
            return 0;  /* Unknown spec */
    }
}

/* ========================================================================
 * Save Checkpoint
 * ======================================================================== */

svmix_status_t svmix_save_checkpoint(const svmix_t* svmix, const char* filepath) {
    if (!svmix || !filepath) {
        return SVMIX_ERR_NULL_POINTER;
    }

    /* Open file for writing */
    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        return SVMIX_ERR_FILE_IO;
    }

    svmix_status_t status = SVMIX_OK;

    /* ====================================================================
     * Write Header (64 bytes)
     * ==================================================================== */

    checkpoint_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));

    /* Magic bytes */
    memcpy(hdr.magic, CHECKPOINT_MAGIC, CHECKPOINT_MAGIC_LEN);

    /* Versions */
    hdr.format_version = CHECKPOINT_FORMAT_VERSION;
    hdr.svmix_major = SVMIX_VERSION_MAJOR;
    hdr.svmix_minor = SVMIX_VERSION_MINOR;
    hdr.svmix_patch = SVMIX_VERSION_PATCH;

    /* Spec and dimensions */
    hdr.spec = (uint32_t)svmix->config.spec;
    hdr.K = (uint32_t)svmix->config.num_models;
    hdr.N = (uint32_t)svmix->config.num_particles;

    /* Timestamp */
    hdr.timestamp = (uint64_t)time(NULL);

    if (fwrite(&hdr, sizeof(hdr), 1, fp) != 1) {
        status = SVMIX_ERR_FILE_IO;
        goto cleanup;
    }

    /* ====================================================================
     * Write Ensemble Metadata
     * ==================================================================== */

    ensemble_t* ens = svmix->ensemble;
    size_t K = ens->K;

    /* Write timestep */
    uint64_t timestep = (uint64_t)ens->timestep;
    if (fwrite(&timestep, sizeof(timestep), 1, fp) != 1) {
        status = SVMIX_ERR_FILE_IO;
        goto cleanup;
    }

    /* Write config (lambda, beta, epsilon, num_threads) */
    /* Zero padding bytes to avoid valgrind warnings */
    ensemble_config_t config_copy;
    memset(&config_copy, 0, sizeof(config_copy));
    config_copy.lambda = ens->config.lambda;
    config_copy.beta = ens->config.beta;
    config_copy.epsilon = ens->config.epsilon;
    config_copy.num_threads = ens->config.num_threads;
    if (fwrite(&config_copy, sizeof(ensemble_config_t), 1, fp) != 1) {
        status = SVMIX_ERR_FILE_IO;
        goto cleanup;
    }

    /* Write weights */
    if (fwrite(ens->weights, sizeof(double), K, fp) != K) {
        status = SVMIX_ERR_FILE_IO;
        goto cleanup;
    }

    /* Write scores */
    for (size_t i = 0; i < K; i++) {
        double score = ens->models[i].score;
        if (fwrite(&score, sizeof(double), 1, fp) != 1) {
            status = SVMIX_ERR_FILE_IO;
            goto cleanup;
        }
    }

    /* ====================================================================
     * Write Per-Model Data (× K)
     * ==================================================================== */

    size_t param_size = get_param_size(svmix->config.spec);
    if (param_size == 0) {
        status = SVMIX_ERR_INVALID_PARAM;
        goto cleanup;
    }

    for (size_t i = 0; i < K; i++) {
        ensemble_model_t* model = &ens->models[i];

        /* Write parameter block size (for forward compatibility) */
        uint32_t size_marker = (uint32_t)param_size;
        if (fwrite(&size_marker, sizeof(size_marker), 1, fp) != 1) {
            status = SVMIX_ERR_FILE_IO;
            goto cleanup;
        }

        /* Write SV parameters (direct struct access!) */
        sv_model_ctx_t* ctx = model->sv_ctx;
        if (fwrite(ctx, param_size, 1, fp) != 1) {
            status = SVMIX_ERR_FILE_IO;
            goto cleanup;
        }

        /* Write fastpf checkpoint blob */
        fastpf_t* pf = &model->pf;
        size_t blob_size = fastpf_checkpoint_bytes(pf);
        if (blob_size == 0) {
            status = SVMIX_ERR_INTERNAL;
            goto cleanup;
        }

        /* Write blob size */
        uint64_t blob_size_marker = (uint64_t)blob_size;
        if (fwrite(&blob_size_marker, sizeof(blob_size_marker), 1, fp) != 1) {
            status = SVMIX_ERR_FILE_IO;
            goto cleanup;
        }

        /* Allocate and write blob */
        void* blob = malloc(blob_size);
        if (!blob) {
            status = SVMIX_ERR_ALLOC_FAILED;
            goto cleanup;
        }

        int fastpf_status = fastpf_checkpoint_write(pf, blob, blob_size);
        if (fastpf_status != FASTPF_SUCCESS) {
            free(blob);
            status = SVMIX_ERR_INTERNAL;
            goto cleanup;
        }

        size_t written = fwrite(blob, 1, blob_size, fp);
        free(blob);

        if (written != blob_size) {
            status = SVMIX_ERR_FILE_IO;
            goto cleanup;
        }
    }

cleanup:
    fclose(fp);
    return status;
}

/* ========================================================================
 * Load Checkpoint
 * ======================================================================== */

svmix_t* svmix_load_checkpoint(const char* filepath, svmix_status_t* status_out) {
    if (!filepath) {
        if (status_out) *status_out = SVMIX_ERR_NULL_POINTER;
        return NULL;
    }

    /* Open file for reading */
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        if (status_out) *status_out = SVMIX_ERR_FILE_IO;
        return NULL;
    }

    svmix_status_t status = SVMIX_OK;
    svmix_t* svmix = NULL;
    ensemble_t* ens = NULL;
    sv_params_t* sv_params = NULL;
    unsigned long* seeds = NULL;
    double* weights = NULL;
    double* scores = NULL;

    /* ====================================================================
     * Read and Validate Header
     * ==================================================================== */

    checkpoint_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        status = SVMIX_ERR_FILE_IO;
        goto cleanup;
    }

    /* Validate magic bytes */
    if (memcmp(hdr.magic, CHECKPOINT_MAGIC, CHECKPOINT_MAGIC_LEN) != 0) {
        status = SVMIX_ERR_CHECKPOINT_CORRUPT;
        goto cleanup;
    }

    /* Check format version */
    if (hdr.format_version > CHECKPOINT_FORMAT_VERSION) {
        status = SVMIX_ERR_VERSION_MISMATCH;
        goto cleanup;
    }

    /* Validate dimensions */
    if (hdr.K == 0 || hdr.N == 0) {
        status = SVMIX_ERR_CHECKPOINT_CORRUPT;
        goto cleanup;
    }

    /* Check if spec is known (V1: only SVMIX_SPEC_VOL supported) */
    if (get_param_size((svmix_spec_t)hdr.spec) == 0) {
        status = SVMIX_ERR_CHECKPOINT_CORRUPT;
        goto cleanup;
    }

    size_t K = (size_t)hdr.K;
    size_t N = (size_t)hdr.N;
    svmix_spec_t spec = (svmix_spec_t)hdr.spec;

    /* ====================================================================
     * Read Ensemble Metadata
     * ==================================================================== */

    uint64_t timestep_marker;
    if (fread(&timestep_marker, sizeof(timestep_marker), 1, fp) != 1) {
        status = SVMIX_ERR_FILE_IO;
        goto cleanup;
    }
    size_t timestep = (size_t)timestep_marker;

    ensemble_config_t ens_config;
    if (fread(&ens_config, sizeof(ens_config), 1, fp) != 1) {
        status = SVMIX_ERR_FILE_IO;
        goto cleanup;
    }

    /* Read weights */
    weights = (double*)malloc(K * sizeof(double));
    if (!weights) {
        status = SVMIX_ERR_ALLOC_FAILED;
        goto cleanup;
    }

    if (fread(weights, sizeof(double), K, fp) != K) {
        free(weights);
        status = SVMIX_ERR_FILE_IO;
        goto cleanup;
    }

    /* Validate weights */
    for (size_t i = 0; i < K; i++) {
        if (!isfinite(weights[i]) || weights[i] < 0.0) {
            free(weights);
            status = SVMIX_ERR_CHECKPOINT_CORRUPT;
            goto cleanup;
        }
    }

    /* Read scores */
    scores = (double*)malloc(K * sizeof(double));
    if (!scores) {
        free(weights);
        status = SVMIX_ERR_ALLOC_FAILED;
        goto cleanup;
    }

    if (fread(scores, sizeof(double), K, fp) != K) {
        free(weights);
        free(scores);
        status = SVMIX_ERR_FILE_IO;
        goto cleanup;
    }

    /* Validate scores */
    for (size_t i = 0; i < K; i++) {
        if (!isfinite(scores[i])) {
            free(weights);
            free(scores);
            status = SVMIX_ERR_CHECKPOINT_CORRUPT;
            goto cleanup;
        }
    }

    /* ====================================================================
     * Allocate svmix_t structure
     * ==================================================================== */

    svmix = (svmix_t*)malloc(sizeof(svmix_t));
    if (!svmix) {
        free(weights);
        free(scores);
        status = SVMIX_ERR_ALLOC_FAILED;
        goto cleanup;
    }

    /* Fill in config */
    svmix->config.num_models = K;
    svmix->config.num_particles = N;
    svmix->config.spec = spec;
    /* Copy ensemble config fields */
    svmix->config.ensemble.lambda = ens_config.lambda;
    svmix->config.ensemble.beta = ens_config.beta;
    svmix->config.ensemble.epsilon = ens_config.epsilon;
    svmix->config.ensemble.num_threads = ens_config.num_threads;
    svmix->timestep = timestep;

    /* Allocate SV params array */
    sv_params = (sv_params_t*)malloc(K * sizeof(sv_params_t));
    if (!sv_params) {
        free(weights);
        free(scores);
        status = SVMIX_ERR_ALLOC_FAILED;
        goto cleanup;
    }
    svmix->sv_params = sv_params;

    /* Allocate seeds array (will be dummy values - we don't save seeds) */
    seeds = (unsigned long*)malloc(K * sizeof(unsigned long));
    if (!seeds) {
        free(weights);
        free(scores);
        status = SVMIX_ERR_ALLOC_FAILED;
        goto cleanup;
    }
    /* Note: Seeds are not preserved across checkpoint - determinism comes from RNG state */
    memset(seeds, 0, K * sizeof(unsigned long));
    svmix->seeds = seeds;

    /* ====================================================================
     * Allocate and populate ensemble
     * ==================================================================== */

    ens = (ensemble_t*)malloc(sizeof(ensemble_t));
    if (!ens) {
        free(weights);
        free(scores);
        status = SVMIX_ERR_ALLOC_FAILED;
        goto cleanup;
    }

    ens->K = K;
    ens->config = ens_config;
    ens->timestep = timestep;
    ens->weights = weights;  /* Transfer ownership */
    ens->models = NULL;

    /* Allocate models array */
    ens->models = (ensemble_model_t*)calloc(K, sizeof(ensemble_model_t));
    if (!ens->models) {
        status = SVMIX_ERR_ALLOC_FAILED;
        goto cleanup;
    }

    /* ====================================================================
     * Read Per-Model Data (× K)
     * ==================================================================== */

    size_t param_size = get_param_size(spec);

    for (size_t i = 0; i < K; i++) {
        ensemble_model_t* model = &ens->models[i];

        /* Read parameter block size */
        uint32_t param_size_marker;
        if (fread(&param_size_marker, sizeof(param_size_marker), 1, fp) != 1) {
            status = SVMIX_ERR_FILE_IO;
            goto cleanup;
        }

        /* Validate parameter size matches spec */
        if (param_size_marker != param_size) {
            status = SVMIX_ERR_CHECKPOINT_CORRUPT;
            goto cleanup;
        }

        /* Read SV parameters into temporary struct */
        sv_model_ctx_t temp_ctx;
        if (fread(&temp_ctx, param_size, 1, fp) != 1) {
            status = SVMIX_ERR_FILE_IO;
            goto cleanup;
        }

        /* Create SV model context and get fastpf callbacks */
        /* NOTE: sv_model_create allocates and validates the context */
        fastpf_model_t fastpf_model;
        sv_model_ctx_t* ctx = sv_model_create(
            temp_ctx.mu_h,
            temp_ctx.phi_h,
            temp_ctx.sigma_h,
            temp_ctx.nu,
            &fastpf_model
        );
        
        if (!ctx) {
            /* sv_model_create failed (invalid parameters) */
            status = SVMIX_ERR_INVALID_PARAM;
            goto cleanup;
        }
        
        /* Assign to model - ownership transferred */
        model->sv_ctx = ctx;

        /* Copy to sv_params array */
        sv_params[i].mu_h = ctx->mu_h;
        sv_params[i].phi_h = ctx->phi_h;
        sv_params[i].sigma_h = ctx->sigma_h;
        sv_params[i].nu = ctx->nu;

        /* Set score */
        model->score = scores[i];
        
        /* Initialize model->pf with zero state */
        memset(&model->pf, 0, sizeof(fastpf_t));
        model->pf.model = fastpf_model;  /* CRITICAL: Set callbacks before restore! */

        /* Read fastpf blob size */
        uint64_t blob_size_marker;
        if (fread(&blob_size_marker, sizeof(blob_size_marker), 1, fp) != 1) {
            status = SVMIX_ERR_FILE_IO;
            goto cleanup;
        }

        /* Read fastpf blob */
        void* blob = malloc(blob_size_marker);
        if (!blob) {
            status = SVMIX_ERR_ALLOC_FAILED;
            goto cleanup;
        }

        if (fread(blob, 1, blob_size_marker, fp) != blob_size_marker) {
            free(blob);
            status = SVMIX_ERR_FILE_IO;
            goto cleanup;
        }

        /* Restore fastpf state (pattern 1: load into uninitialized pf) */
        /* Note: fastpf_checkpoint_read requires cfg when pf is uninitialized */
        fastpf_cfg_t pf_cfg;
        fastpf_cfg_init(&pf_cfg, N, sizeof(double));
        pf_cfg.rng_seed = 0;  /* Will be overwritten from checkpoint */
        pf_cfg.resample_threshold = 0.5;
        pf_cfg.num_threads = ens_config.num_threads;
        
        int fastpf_status = fastpf_checkpoint_read(&model->pf, &pf_cfg, blob, blob_size_marker);
        free(blob);

        if (fastpf_status != FASTPF_SUCCESS) {
            /* Map fastpf errors to svmix errors */
            switch (fastpf_status) {
                case FASTPF_ERR_CHECKPOINT_MAGIC:
                case FASTPF_ERR_CHECKPOINT_VERSION:
                case FASTPF_ERR_CHECKPOINT_PORTABILITY:
                case FASTPF_ERR_CHECKPOINT_SIZE:
                case FASTPF_ERR_CHECKPOINT_CORRUPT:
                    status = SVMIX_ERR_CHECKPOINT_CORRUPT;
                    break;
                case FASTPF_ERR_INVALID_ARG:
                    status = SVMIX_ERR_NULL_POINTER;
                    break;
                case FASTPF_ERR_ALLOC:
                    status = SVMIX_ERR_ALLOC_FAILED;
                    break;
                default:
                    status = SVMIX_ERR_INTERNAL;
                    break;
            }
            goto cleanup;
        }
    }

    /* Free temporary scores array */
    free(scores);
    scores = NULL;

    /* Link ensemble to svmix */
    svmix->ensemble = ens;

    /* Success! */
    fclose(fp);
    if (status_out) *status_out = SVMIX_OK;
    return svmix;

cleanup:
    /* Cleanup on error */
    fclose(fp);

    if (ens) {
        if (ens->models) {
            for (size_t i = 0; i < K; i++) {
                if (ens->models[i].sv_ctx) {
                    free(ens->models[i].sv_ctx);
                }
                /* fastpf cleanup handled by fastpf_checkpoint_read on failure */
            }
            free(ens->models);
        }
        if (ens->weights) {
            free(ens->weights);
        }
        free(ens);
    }

    if (scores) free(scores);

    if (svmix) {
        if (sv_params) free(sv_params);
        if (seeds) free(seeds);
        free(svmix);
    }

    if (status_out) *status_out = status;
    return NULL;
}
