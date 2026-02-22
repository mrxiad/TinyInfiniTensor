.PHONY : build clean format install-python test-cpp test-onnx

TYPE ?= Release
TEST ?= ON
C_COMPILER ?= clang-14
CXX_COMPILER ?= clang++-14
JOBS ?= 8
BUILD_DIR ?= build/clang14-$(TYPE)

CMAKE_OPT = -DCMAKE_BUILD_TYPE=$(TYPE)
CMAKE_OPT += -DBUILD_TEST=$(TEST)
CMAKE_OPT += -DCMAKE_C_COMPILER=$(C_COMPILER)
CMAKE_OPT += -DCMAKE_CXX_COMPILER=$(CXX_COMPILER)

build:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake $(CMAKE_OPT) ../.. && make -j$(JOBS)

clean:
	rm -rf build

test-cpp:
	@echo
	cd $(BUILD_DIR) && make test
