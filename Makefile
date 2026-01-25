# ========================================================================
# svmix — Mixture-of-models Bayesian Belief Engine for SV Inference
# ========================================================================

# Compiler and flags
CC := gcc
CFLAGS := -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror
LDFLAGS := -lm

# Directories
SRC_DIR := src
INCLUDE_DIR := include
TEST_DIR := tests
FASTPF_DIR := third_party/fastpf
BIN_DIR := bin

# fastpf library
FASTPF_LIB := $(FASTPF_DIR)/bin/libfastpf.a

# Source files
SVMIX_SRCS := $(SRC_DIR)/util.c $(SRC_DIR)/model_sv.c $(SRC_DIR)/ensemble.c
SVMIX_OBJS := $(SVMIX_SRCS:$(SRC_DIR)/%.c=$(BIN_DIR)/%.o)

# Test files
TEST_STUDENT_T := $(TEST_DIR)/unit/test_student_t.c
TEST_MODEL_SV := $(TEST_DIR)/unit/test_model_sv.c
TEST_ENSEMBLE := $(TEST_DIR)/unit/test_ensemble.c
TEST_SV_FASTPF_SMOKE := $(TEST_DIR)/integration/test_sv_fastpf_smoke.c
TEST_SV_FASTPF_DETERMINISM := $(TEST_DIR)/integration/test_sv_fastpf_determinism.c

# Targets
.PHONY: all clean test test-unit test-integration test-student-t test-model-sv test-ensemble test-smoke test-determinism fastpf

all: $(BIN_DIR)/test_student_t $(BIN_DIR)/test_model_sv $(BIN_DIR)/test_ensemble $(BIN_DIR)/test_sv_fastpf_smoke $(BIN_DIR)/test_sv_fastpf_determinism

# Build fastpf library
fastpf: $(FASTPF_LIB)

$(FASTPF_LIB):
	@if [ ! -f "$(FASTPF_DIR)/Makefile" ]; then \
		echo "ERROR: fastpf submodule not initialized!"; \
		echo "Run: git submodule update --init --recursive"; \
		exit 1; \
	fi
	@echo "Building fastpf..."
	@$(MAKE) -C $(FASTPF_DIR) -s

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build object files
$(BIN_DIR)/%.o: $(SRC_DIR)/%.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(FASTPF_DIR)/include -c $< -o $@

# Build test_student_t
$(BIN_DIR)/test_student_t: $(TEST_STUDENT_T) $(SVMIX_OBJS) $(FASTPF_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(FASTPF_DIR)/include $(TEST_STUDENT_T) $(SVMIX_OBJS) $(FASTPF_LIB) -o $@ $(LDFLAGS)

# Build test_model_sv
$(BIN_DIR)/test_model_sv: $(TEST_MODEL_SV) $(SVMIX_OBJS) $(FASTPF_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(FASTPF_DIR)/include $(TEST_MODEL_SV) $(SVMIX_OBJS) $(FASTPF_LIB) -o $@ $(LDFLAGS)

# Build test_ensemble
$(BIN_DIR)/test_ensemble: $(TEST_ENSEMBLE) $(SVMIX_OBJS) $(FASTPF_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(FASTPF_DIR)/include $(TEST_ENSEMBLE) $(SVMIX_OBJS) $(FASTPF_LIB) -o $@ $(LDFLAGS)

# Build test_sv_fastpf_smoke
$(BIN_DIR)/test_sv_fastpf_smoke: $(TEST_SV_FASTPF_SMOKE) $(SVMIX_OBJS) $(FASTPF_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(FASTPF_DIR)/include $(TEST_SV_FASTPF_SMOKE) $(SVMIX_OBJS) $(FASTPF_LIB) -o $@ $(LDFLAGS)

# Build test_sv_fastpf_determinism
$(BIN_DIR)/test_sv_fastpf_determinism: $(TEST_SV_FASTPF_DETERMINISM) $(SVMIX_OBJS) $(FASTPF_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(FASTPF_DIR)/include $(TEST_SV_FASTPF_DETERMINISM) $(SVMIX_OBJS) $(FASTPF_LIB) -o $@ $(LDFLAGS)

# Run tests
test-student-t: $(BIN_DIR)/test_student_t
	@./$(BIN_DIR)/test_student_t

test-model-sv: $(BIN_DIR)/test_model_sv
	@./$(BIN_DIR)/test_model_sv

test-ensemble: $(BIN_DIR)/test_ensemble
	@./$(BIN_DIR)/test_ensemble

test-smoke: $(BIN_DIR)/test_sv_fastpf_smoke
	@./$(BIN_DIR)/test_sv_fastpf_smoke

test-determinism: $(BIN_DIR)/test_sv_fastpf_determinism
	@./$(BIN_DIR)/test_sv_fastpf_determinism

test-unit: test-student-t test-model-sv test-ensemble

test-integration: test-smoke test-determinism

test: test-unit test-integration

# Clean
clean:
	rm -rf $(BIN_DIR)

# Help
help:
	@echo "svmix Makefile targets:"
	@echo "  all              - Build all test binaries"
	@echo "  test             - Run all tests (unit + integration)"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-student-t   - Run Student-t log-PDF tests"
	@echo "  test-model-sv    - Run SV model callback tests"
	@echo "  test-ensemble    - Run ensemble tests"
	@echo "  test-smoke       - Run SV+fastpf smoke tests"
	@echo "  test-determinism - Run determinism tests"
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help message"
