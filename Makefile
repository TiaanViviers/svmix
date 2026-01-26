CC := gcc
CFLAGS := -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror
LDFLAGS := -lm

# Optional: Enable OpenMP for parallel ensemble stepping
# Usage: make OPENMP=1 test
ifdef OPENMP
    CFLAGS += -fopenmp
    LDFLAGS += -fopenmp
endif

# Directories
SRC_DIR := src
INCLUDE_DIR := include
TEST_DIR := tests
EXAMPLE_DIR := examples
FASTPF_DIR := third_party/fastpf
BIN_DIR := bin

# fastpf library
FASTPF_LIB := $(FASTPF_DIR)/bin/libfastpf.a

# Source files
SVMIX_SRCS := $(SRC_DIR)/util.c $(SRC_DIR)/model_sv.c $(SRC_DIR)/ensemble.c $(SRC_DIR)/svmix.c $(SRC_DIR)/checkpoint.c
SVMIX_OBJS := $(SVMIX_SRCS:$(SRC_DIR)/%.c=$(BIN_DIR)/%.o)

# Test files
TEST_STUDENT_T := $(TEST_DIR)/unit/test_student_t.c
TEST_MODEL_SV := $(TEST_DIR)/unit/test_model_sv.c
TEST_ENSEMBLE := $(TEST_DIR)/unit/test_ensemble.c
TEST_ENSEMBLE_OPENMP := $(TEST_DIR)/unit/test_ensemble_openmp.c
TEST_SVMIX_API := $(TEST_DIR)/unit/test_svmix_api.c
TEST_CHECKPOINT := $(TEST_DIR)/unit/test_checkpoint.c
TEST_SV_FASTPF_SMOKE := $(TEST_DIR)/integration/test_sv_fastpf_smoke.c
TEST_SV_FASTPF_DETERMINISM := $(TEST_DIR)/integration/test_sv_fastpf_determinism.c

# Example files
EXAMPLE_BASIC := $(EXAMPLE_DIR)/example_basic_usage.c

# Targets
.PHONY: all clean test test-unit test-integration test-student-t test-model-sv test-ensemble test-ensemble-openmp test-svmix-api test-checkpoint test-smoke test-determinism fastpf examples

all: $(BIN_DIR)/test_student_t $(BIN_DIR)/test_model_sv $(BIN_DIR)/test_ensemble $(BIN_DIR)/test_ensemble_openmp $(BIN_DIR)/test_svmix_api $(BIN_DIR)/test_checkpoint $(BIN_DIR)/test_sv_fastpf_smoke $(BIN_DIR)/test_sv_fastpf_determinism

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

# Build test_ensemble_openmp (only useful with OPENMP=1)
$(BIN_DIR)/test_ensemble_openmp: $(TEST_ENSEMBLE_OPENMP) $(SVMIX_OBJS) $(FASTPF_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(FASTPF_DIR)/include $(TEST_ENSEMBLE_OPENMP) $(SVMIX_OBJS) $(FASTPF_LIB) -o $@ $(LDFLAGS)

# Build test_svmix_api
$(BIN_DIR)/test_svmix_api: $(TEST_SVMIX_API) $(SVMIX_OBJS) $(FASTPF_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(FASTPF_DIR)/include $(TEST_SVMIX_API) $(SVMIX_OBJS) $(FASTPF_LIB) -o $@ $(LDFLAGS)

# Build test_checkpoint
$(BIN_DIR)/test_checkpoint: $(TEST_CHECKPOINT) $(SVMIX_OBJS) $(FASTPF_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(FASTPF_DIR)/include $(TEST_CHECKPOINT) $(SVMIX_OBJS) $(FASTPF_LIB) -o $@ $(LDFLAGS)

# Build test_sv_fastpf_smoke
$(BIN_DIR)/test_sv_fastpf_smoke: $(TEST_SV_FASTPF_SMOKE) $(SVMIX_OBJS) $(FASTPF_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(FASTPF_DIR)/include $(TEST_SV_FASTPF_SMOKE) $(SVMIX_OBJS) $(FASTPF_LIB) -o $@ $(LDFLAGS)

# Build test_sv_fastpf_determinism
$(BIN_DIR)/test_sv_fastpf_determinism: $(TEST_SV_FASTPF_DETERMINISM) $(SVMIX_OBJS) $(FASTPF_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(FASTPF_DIR)/include $(TEST_SV_FASTPF_DETERMINISM) $(SVMIX_OBJS) $(FASTPF_LIB) -o $@ $(LDFLAGS)

# Build examples
examples: $(BIN_DIR)/example_basic_usage

$(BIN_DIR)/example_basic_usage: $(EXAMPLE_BASIC) $(SVMIX_OBJS) $(FASTPF_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -I$(FASTPF_DIR)/include $(EXAMPLE_BASIC) $(SVMIX_OBJS) $(FASTPF_LIB) -o $@ $(LDFLAGS)

# Run tests
test-student-t: $(BIN_DIR)/test_student_t
	@./$(BIN_DIR)/test_student_t

test-model-sv: $(BIN_DIR)/test_model_sv
	@./$(BIN_DIR)/test_model_sv

test-ensemble: $(BIN_DIR)/test_ensemble
	@./$(BIN_DIR)/test_ensemble

test-ensemble-openmp: $(BIN_DIR)/test_ensemble_openmp
	@./$(BIN_DIR)/test_ensemble_openmp

test-svmix-api: $(BIN_DIR)/test_svmix_api
	@./$(BIN_DIR)/test_svmix_api

test-checkpoint: $(BIN_DIR)/test_checkpoint
	@./$(BIN_DIR)/test_checkpoint

test-smoke: $(BIN_DIR)/test_sv_fastpf_smoke
	@./$(BIN_DIR)/test_sv_fastpf_smoke

test-determinism: $(BIN_DIR)/test_sv_fastpf_determinism
	@./$(BIN_DIR)/test_sv_fastpf_determinism

test-unit: test-student-t test-model-sv test-ensemble test-ensemble-openmp test-svmix-api test-checkpoint

test-integration: test-smoke test-determinism

test: test-unit test-integration

# =============================================================================
# Memory checking targets
# =============================================================================

# All test binaries to check
TEST_BINS := $(BIN_DIR)/test_student_t \
             $(BIN_DIR)/test_model_sv \
             $(BIN_DIR)/test_ensemble \
             $(BIN_DIR)/test_ensemble_openmp \
             $(BIN_DIR)/test_svmix_api \
             $(BIN_DIR)/test_checkpoint \
             $(BIN_DIR)/test_sv_fastpf_smoke \
             $(BIN_DIR)/test_sv_fastpf_determinism

# Example binaries to check
EXAMPLE_BINS := $(BIN_DIR)/example_basic_usage

# All binaries to check
ALL_CHECK_BINS := $(TEST_BINS) $(EXAMPLE_BINS)

# Valgrind flags
VALGRIND := valgrind
VALGRIND_FLAGS := --leak-check=full \
                  --show-leak-kinds=all \
                  --track-origins=yes \
                  --verbose \
                  --error-exitcode=1 \
                  --suppressions=.valgrind.supp

# Sanitizer flags (separate builds)
ASAN_FLAGS := -fsanitize=address -fno-omit-frame-pointer -g
UBSAN_FLAGS := -fsanitize=undefined -fno-omit-frame-pointer -g
MSAN_FLAGS := -fsanitize=memory -fno-omit-frame-pointer -g

.PHONY: valgrind sanitizer asan ubsan check-memory

# Run valgrind on all tests and examples
# Note: Rebuilds without sanitizers to avoid ASan+Valgrind conflict
valgrind: clean all examples .valgrind.supp
	@echo "=========================================="
	@echo "Running Valgrind on all tests and examples"
	@echo "=========================================="
	@for bin in $(ALL_CHECK_BINS); do \
		echo ""; \
		echo "Checking $$bin..."; \
		$(VALGRIND) $(VALGRIND_FLAGS) ./$$bin || exit 1; \
	done
	@echo ""
	@echo "=========================================="
	@echo "Valgrind: All checks passed!"
	@echo "=========================================="

# Create valgrind suppressions file if it doesn't exist
.valgrind.supp:
	@echo "Creating valgrind suppressions file..."
	@echo "# Valgrind suppressions for svmix" > .valgrind.supp
	@echo "# Add suppressions here if needed" >> .valgrind.supp
	@echo "" >> .valgrind.supp
	@echo "# OpenMP false positives (if OPENMP=1)" >> .valgrind.supp
	@echo "{" >> .valgrind.supp
	@echo "   openmp_thread_leak" >> .valgrind.supp
	@echo "   Memcheck:Leak" >> .valgrind.supp
	@echo "   ..." >> .valgrind.supp
	@echo "   obj:*/libgomp.so*" >> .valgrind.supp
	@echo "}" >> .valgrind.supp

# Run AddressSanitizer (detects memory errors)
asan: CFLAGS += $(ASAN_FLAGS)
asan: LDFLAGS += $(ASAN_FLAGS)
asan: clean all examples
	@echo "=========================================="
	@echo "Running AddressSanitizer on all tests and examples"
	@echo "=========================================="
	@for bin in $(ALL_CHECK_BINS); do \
		echo ""; \
		echo "Checking $$bin..."; \
		ASAN_OPTIONS=detect_leaks=1:check_initialization_order=1 ./$$bin || exit 1; \
	done
	@echo ""
	@echo "=========================================="
	@echo "AddressSanitizer: All checks passed!"
	@echo "=========================================="

# Run UndefinedBehaviorSanitizer (detects undefined behavior)
ubsan: CFLAGS += $(UBSAN_FLAGS)
ubsan: LDFLAGS += $(UBSAN_FLAGS)
ubsan: clean all examples
	@echo "=========================================="
	@echo "Running UndefinedBehaviorSanitizer on all tests and examples"
	@echo "=========================================="
	@for bin in $(ALL_CHECK_BINS); do \
		echo ""; \
		echo "Checking $$bin..."; \
		UBSAN_OPTIONS=print_stacktrace=1 ./$$bin || exit 1; \
	done
	@echo ""
	@echo "=========================================="
	@echo "UndefinedBehaviorSanitizer: All checks passed!"
	@echo "=========================================="

# Run all sanitizers (ASan + UBSan together)
sanitizer: CFLAGS += $(ASAN_FLAGS) $(UBSAN_FLAGS)
sanitizer: LDFLAGS += $(ASAN_FLAGS) $(UBSAN_FLAGS)
sanitizer: clean all examples
	@echo "=========================================="
	@echo "Running Sanitizers (ASan + UBSan) on all tests and examples"
	@echo "=========================================="
	@for bin in $(ALL_CHECK_BINS); do \
		echo ""; \
		echo "Checking $$bin..."; \
		ASAN_OPTIONS=detect_leaks=1:check_initialization_order=1 \
		UBSAN_OPTIONS=print_stacktrace=1 \
		./$$bin || exit 1; \
	done
	@echo ""
	@echo "=========================================="
	@echo "Sanitizers: All checks passed!"
	@echo "=========================================="

# Run all memory checks (valgrind + sanitizers)
check-memory: valgrind sanitizer
	@echo ""
	@echo "=========================================="
	@echo "All memory checks passed!"
	@echo "=========================================="

# Clean
clean:
	rm -rf $(BIN_DIR)
	rm -f .valgrind.supp *.valgrind.supp
	@find . -maxdepth 1 -type f \( \
		-name "vgcore.*" -o \
		-name "callgrind.out.*" -o \
		-name "cachegrind.out.*" -o \
		-name "massif.out.*" -o \
		-name "helgrind.out.*" -o \
		-name "drd.out.*" \
	\) -delete

# Help
help:
	@echo "svmix Makefile targets:"
	@echo ""
	@echo "Build targets:"
	@echo "  all              - Build all test binaries"
	@echo "  examples         - Build example programs"
	@echo ""
	@echo "Test targets:"
	@echo "  test             - Run all tests (unit + integration)"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-student-t   - Run Student-t log-PDF tests"
	@echo "  test-model-sv    - Run SV model callback tests"
	@echo "  test-ensemble    - Run ensemble tests"
	@echo "  test-ensemble-openmp - Run OpenMP-specific tests (requires OPENMP=1)"
	@echo "  test-svmix-api   - Run public API tests"
	@echo "  test-smoke       - Run SV+fastpf smoke tests"
	@echo "  test-determinism - Run determinism tests"
	@echo ""
	@echo "Memory checking targets:"
	@echo "  valgrind         - Run valgrind on all tests and examples"
	@echo "  asan             - Run AddressSanitizer on all tests and examples"
	@echo "  ubsan            - Run UndefinedBehaviorSanitizer on all tests and examples"
	@echo "  sanitizer        - Run ASan + UBSan together on all tests and examples"
	@echo "  check-memory     - Run all memory checks (valgrind + sanitizers)"
	@echo ""
	@echo "Other targets:"
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  OPENMP=1         - Enable OpenMP for parallel ensemble stepping"
	@echo "                     Example: make OPENMP=1 test-ensemble"
