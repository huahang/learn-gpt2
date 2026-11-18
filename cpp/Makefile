# learn-gpt2 C++ 推理框架.
#
#   make                 构建两个后端的 cubin 和 C++ 程序 build/bin/gpt2
#   make kernels         只构建 cubin: build/kernels/<后端>/
#   make bin             只构建 C++ 程序
#   make check           检查 cubin 接口, 运行 kernel 与端到端测试
#   make run MODEL=...   用 GPT-2 生成一段文本
#   make clean           删除 build/
#
# 变量:
#   ARCH=sm_120a         GPU 架构. 默认读取第 0 号 GPU 的计算能力; SM90 及更新的架构加后缀 a.
#   BACKENDS="tilelang cute_dsl"   要构建的 kernel 后端.
#   CUDA_HOME=/usr/local/cuda      提供 cuda.h, libcuda 的 stub 和 nvcc (TileLang 后端需要).
#   PYTHON="uv run --extra cuda python"   运行 kernel 构建脚本的 Python.

CUDA_HOME ?= $(patsubst %/bin/nvcc,%,$(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc))
PYTHON ?= uv run --extra cuda python
BACKENDS ?= tilelang cute_dsl
ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | \
          awk -F. 'NR == 1 { printf "sm_%d%d%s", $$1, $$2, ($$1 >= 9 ? "a" : "") }')

BUILD := build
OBJ := $(BUILD)/obj
BIN := $(BUILD)/bin/gpt2
KERNELS := $(BUILD)/kernels

CXX ?= g++
# C++20, 需要 std::format: GCC 13 或 Clang 17 及更新的版本.
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++20 -Wall -Wextra -MMD -MP -I$(CUDA_HOME)/include -Ithird_party/safetensors-cpp
# 只链接 driver API (libcuda). stub 目录使没有驱动的机器也能链接; 运行时加载驱动提供的 libcuda.so.1.
LDFLAGS += -L$(CUDA_HOME)/lib64/stubs -L$(CUDA_HOME)/lib64
LDLIBS += -lcuda

SOURCES := $(wildcard src/*.cc)
OBJECTS := $(SOURCES:src/%.cc=$(OBJ)/%.o) $(OBJ)/safetensors.o
MANIFESTS := $(BACKENDS:%=$(KERNELS)/%/manifest.txt)

.PHONY: all bin kernels check-abi check run clean FORCE

all: kernels bin

bin: $(BIN)

kernels: $(MANIFESTS)

$(BIN): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(OBJ)/%.o: src/%.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# 第三方代码不显示警告.
$(OBJ)/safetensors.o: third_party/safetensors-cpp/safetensors.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -w -c -o $@ $<

# ARCH 改变时重新构建 cubin: arch.txt 的内容只在 ARCH 变化时更新.
$(BUILD)/arch.txt: FORCE
	@test -n "$(ARCH)" || { echo "无法检测 GPU 架构, 请指定 ARCH, 例如 make ARCH=sm_120a" >&2; exit 1; }
	@mkdir -p $(BUILD)
	@echo "$(ARCH)" | cmp -s - $@ || echo "$(ARCH)" > $@

$(KERNELS)/tilelang/manifest.txt: kernels/build_tilelang.py kernels/kernel_abi.py $(BUILD)/arch.txt
	$(PYTHON) kernels/build_tilelang.py --arch $(ARCH) --out $(dir $@)

$(KERNELS)/cute_dsl/manifest.txt: kernels/build_cute_dsl.py kernels/kernel_abi.py \
		../src/learn_gpt2/gpt2_inference_cute_dsl.py $(BUILD)/arch.txt
	$(PYTHON) kernels/build_cute_dsl.py --arch $(ARCH) --out $(dir $@)

check-abi: $(MANIFESTS)
	$(PYTHON) kernels/kernel_abi.py $(dir $(MANIFESTS))

check: all check-abi
	cd .. && uv run --extra cuda pytest tests/test_cpp_kernels.py tests/test_cpp_inference.py

# 例: make run MODEL=~/.cache/huggingface/hub/models--gpt2/snapshots/<hash> BACKEND=cute_dsl
BACKEND ?= tilelang
PROMPT ?= Hello, I am a language model,
run: all
	@test -n "$(MODEL)" || { echo "请指定 MODEL=<safetensors 文件或检查点目录>" >&2; exit 1; }
	$(BIN) --model "$(MODEL)" --backend $(BACKEND) --prompt "$(PROMPT)"

clean:
	rm -rf $(BUILD)

-include $(OBJECTS:.o=.d)
