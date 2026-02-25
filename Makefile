.PHONY : help build configure check-clang clean format install-python test test-cpp test-onnx
.DEFAULT_GOAL := build

TYPE ?= Release
TEST ?= ON
CLANG_VERSION ?= 14
C_COMPILER ?= clang-$(CLANG_VERSION)
CXX_COMPILER ?= clang++-$(CLANG_VERSION)
JOBS ?= 8

LIBSTDCXX_VERSION ?= 9
LIBSTDCXX_ARCH ?= x86_64-linux-gnu
LIBSTDCXX_INCLUDE ?= /usr/include/c++/$(LIBSTDCXX_VERSION)
LIBSTDCXX_ARCH_INCLUDE ?= /usr/include/$(LIBSTDCXX_ARCH)/c++/$(LIBSTDCXX_VERSION)
LIBSTDCXX_BACKWARD_INCLUDE ?= $(LIBSTDCXX_INCLUDE)/backward
LIBSTDCXX_LIBDIR ?= /usr/lib/gcc/$(LIBSTDCXX_ARCH)/$(LIBSTDCXX_VERSION)

BUILD_DIR ?= build/clang$(CLANG_VERSION)-$(TYPE)

CLANG_SYS_INCLUDES = -isystem $(LIBSTDCXX_INCLUDE) \
	-isystem $(LIBSTDCXX_ARCH_INCLUDE) \
	-isystem $(LIBSTDCXX_BACKWARD_INCLUDE)
CLANG_LINK_FLAGS = -L$(LIBSTDCXX_LIBDIR)

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)
CMAKE_OPT += -DBUILD_TEST=$(TEST)
CMAKE_OPT += -DCMAKE_C_COMPILER=$(C_COMPILER)
CMAKE_OPT += -DCMAKE_CXX_COMPILER=$(CXX_COMPILER)
CMAKE_OPT += -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
CMAKE_OPT += -DCMAKE_CXX_FLAGS='$(CLANG_SYS_INCLUDES)'
CMAKE_OPT += -DCMAKE_EXE_LINKER_FLAGS='$(CLANG_LINK_FLAGS)'
CMAKE_OPT += -DCMAKE_SHARED_LINKER_FLAGS='$(CLANG_LINK_FLAGS)'

help:
	@echo "Targets:"
	@echo "  make build          Configure + build with clang"
	@echo "  make configure      Run CMake configure only"
	@echo "  make test           Run C++ tests"
	@echo "  make test-cpp       Run C++ tests"
	@echo "  make format         Format C/C++ sources if clang-format is available"
	@echo "  make clean          Remove build outputs"
	@echo
	@echo "Key variables (override with VAR=value):"
	@echo "  CLANG_VERSION=$(CLANG_VERSION)"
	@echo "  LIBSTDCXX_VERSION=$(LIBSTDCXX_VERSION)"
	@echo "  TYPE=$(TYPE), TEST=$(TEST), JOBS=$(JOBS)"

check-clang:
	@command -v $(C_COMPILER) >/dev/null || (echo "error: $(C_COMPILER) not found"; exit 1)
	@command -v $(CXX_COMPILER) >/dev/null || (echo "error: $(CXX_COMPILER) not found"; exit 1)
	@test -d $(LIBSTDCXX_INCLUDE) || (echo "error: missing $(LIBSTDCXX_INCLUDE), set LIBSTDCXX_VERSION=<version>"; exit 1)
	@test -d $(LIBSTDCXX_ARCH_INCLUDE) || (echo "error: missing $(LIBSTDCXX_ARCH_INCLUDE), set LIBSTDCXX_ARCH/LIBSTDCXX_VERSION"; exit 1)
	@test -d $(LIBSTDCXX_LIBDIR) || (echo "error: missing $(LIBSTDCXX_LIBDIR), set LIBSTDCXX_ARCH/LIBSTDCXX_VERSION"; exit 1)

configure: check-clang
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake $(CMAKE_OPT) ../..
	ln -sfn $(BUILD_DIR)/compile_commands.json compile_commands.json

build: configure
	$(MAKE) -C $(BUILD_DIR) -j$(JOBS)

format:
	@formatter="clang-format-$(CLANG_VERSION)"; \
	if ! command -v $$formatter >/dev/null; then \
		if command -v clang-format >/dev/null; then formatter="clang-format"; \
		else echo "warning: clang-format not found, skip"; exit 0; fi; \
	fi; \
	files="$$(rg --files include src test -g '*.h' -g '*.hpp' -g '*.cc' -g '*.cpp' 2>/dev/null || true)"; \
	if [ -z "$$files" ]; then echo "no C/C++ files found"; exit 0; fi; \
	echo "format with $$formatter"; \
	$$formatter -i $$files

install-python:
	@echo "No Python package install step is defined for this repository. Skipped."

test: test-cpp

test-cpp: build
	@echo
	cd $(BUILD_DIR) && $(MAKE) test

test-onnx:
	@echo "No ONNX tests are configured in this repository. Skipped."

clean:
	rm -rf build compile_commands.json
