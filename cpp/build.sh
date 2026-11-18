#!/usr/bin/env bash
# 构建 learn-gpt2 的 C++ 推理框架: 两个后端的 cubin kernel 与 C++ 程序 build/bin/gpt2.
#
# 用法: cpp/build.sh [选项]
#   --arch sm_XX      GPU 架构, 例如 sm_120a 或 sm_86. 默认读取第 0 号 GPU
#   --backend NAME    只构建一个 kernel 后端: tilelang 或 cute_dsl. 默认两个都构建
#   --jobs N          并行任务数, 默认为 CPU 核数
#   --clean           先删除 build/
#   --check           构建后检查 cubin 接口, 并运行 kernel 测试与端到端测试
#   -h, --help        显示本帮助
#
# 本脚本检查工具后调用 Makefile; 也可以直接使用 make, 见 Makefile 开头的说明.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

arch=""
backends="tilelang cute_dsl"
jobs="$(nproc)"
clean=0
check=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) arch="$2"; shift 2 ;;
    --backend) backends="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    --clean) clean=1; shift ;;
    --check) check=1; shift ;;
    -h | --help) sed -n '2,12p' "$(basename "${BASH_SOURCE[0]}")" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知选项: $1 (见 --help)" >&2; exit 1 ;;
  esac
done
for backend in $backends; do
  if [[ "$backend" != tilelang && "$backend" != cute_dsl ]]; then
    echo "未知的 kernel 后端: $backend. 可选 tilelang 或 cute_dsl" >&2
    exit 1
  fi
done

need() {
  if ! command -v "$1" > /dev/null 2>&1; then
    echo "缺少 $1. $2" >&2
    exit 1
  fi
}
need make "请安装 GNU Make."
need "${CXX:-g++}" "需要支持 C++20 与 std::format 的编译器, 例如 GCC 13 或 Clang 17 及更新的版本."
need uv "kernel 构建脚本在 uv 管理的 Python 环境中运行, 见 https://docs.astral.sh/uv/."
if [[ " $backends " == *" tilelang "* ]]; then
  need nvcc "TileLang 后端用 nvcc 把生成的 CUDA C++ 编译为 cubin. 请安装 CUDA Toolkit."
fi
if [[ -z "$arch" ]]; then
  need nvidia-smi "无法检测 GPU 架构. 请用 --arch 指定, 例如 --arch sm_120a."
fi

make_args=(-j "$jobs" BACKENDS="$backends")
if [[ -n "$arch" ]]; then
  make_args+=(ARCH="$arch")
fi
if [[ $clean == 1 ]]; then
  make clean
fi
if [[ $check == 1 ]]; then
  make "${make_args[@]}" check
else
  make "${make_args[@]}" all
fi
echo "完成. 程序: cpp/build/bin/gpt2; cubin: cpp/build/kernels/{${backends// /,}}/"
