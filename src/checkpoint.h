/**
 * @file checkpoint.h
 * @brief Internal checkpoint format definitions for svmix.
 *
 * File format: Binary, spec-aware, versioned.
 * Extension: .svmix
 *
 * Design principles:
 *   - Spec-aware: Different specs have different param sizes
 *   - Self-describing: Sizes embedded for forward compatibility
 *   - Deterministic: Complete state save/restore
 *   - Validated: Magic bytes, version checks, size checks
 */

#ifndef SVMIX_CHECKPOINT_H
#define SVMIX_CHECKPOINT_H

#include "svmix.h"
#include <stdint.h>
#include <stdio.h>

/* ========================================================================
 * Checkpoint Format Version
 * ======================================================================== */

/**
 * Checkpoint format version (independent of svmix version).
 * Only bump if file structure changes (not for new specs).
 */
#define CHECKPOINT_FORMAT_VERSION  1

/**
 * Magic bytes for checkpoint identification.
 */
#define CHECKPOINT_MAGIC  "SVMIXCP1"
#define CHECKPOINT_MAGIC_LEN  8

/* ========================================================================
 * Checkpoint Header (64 bytes fixed)
 * ======================================================================== */

/**
 * @brief Checkpoint file header.
 *
 * Fixed size (64 bytes) for easy parsing.
 * Contains all metadata needed to validate and parse the checkpoint.
 */
typedef struct {
    char magic[8];              /**< "SVMIXCP1" magic bytes */
    uint32_t format_version;    /**< Checkpoint format version (1) */
    uint32_t svmix_major;       /**< svmix version: major */
    uint32_t svmix_minor;       /**< svmix version: minor */
    uint32_t svmix_patch;       /**< svmix version: patch */
    uint32_t spec;              /**< Model spec (SVMIX_SPEC_VOL, etc.) */
    uint64_t timestamp;         /**< Unix epoch timestamp */
    uint32_t K;                 /**< Number of models */
    uint32_t N;                 /**< Particles per model */
    uint8_t reserved[16];       /**< Reserved for future use */
} checkpoint_header_t;

/* Verify header is exactly 64 bytes at compile time */
_Static_assert(sizeof(checkpoint_header_t) == 64, 
               "checkpoint_header_t must be exactly 64 bytes");

#endif /* SVMIX_CHECKPOINT_H */
