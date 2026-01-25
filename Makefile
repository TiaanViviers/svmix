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

# Source files
SVMIX_SRCS := $(SRC_DIR)/util.c
SVMIX_OBJS := $(SVMIX_SRCS:$(SRC_DIR)/%.c=$(BIN_DIR)/%.o)

# Test files
TEST_STUDENT_T := $(TEST_DIR)/unit/test_student_t.c

# Targets
.PHONY: all clean test test-student-t

all: $(BIN_DIR)/test_student_t

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build object files
$(BIN_DIR)/%.o: $(SRC_DIR)/%.c | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -c $< -o $@

# Build test_student_t
$(BIN_DIR)/test_student_t: $(TEST_STUDENT_T) $(SVMIX_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) $^ -o $@ $(LDFLAGS)

# Run tests
test-student-t: $(BIN_DIR)/test_student_t
	@./$(BIN_DIR)/test_student_t

test: test-student-t

# Clean
clean:
	rm -rf $(BIN_DIR)

# Help
help:
	@echo "svmix Makefile targets:"
	@echo "  all              - Build all test binaries"
	@echo "  test             - Run all tests"
	@echo "  test-student-t   - Run Student-t log-PDF tests"
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help message"
