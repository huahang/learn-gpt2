// 按 manifest.txt 加载一组 cubin kernel, 并按 kernel ABI 启动它们.
//
// 接口定义见 cpp/kernels/kernel_abi.py. TileLang 与 CuTe DSL 两个后端导出相同的接口,
// 所以本类不区分后端: 换一个 cubin 目录, 就换了一个后端.
#pragma once

#include <cuda.h>

#include <filesystem>
#include <map>
#include <memory>
#include <string>

namespace gpt2 {

// ABI 规定的常量, 与 kernel_abi.py 一致.
inline constexpr int kAbiVersion = 1;
inline constexpr int kThreads = 128;
inline constexpr int kTileM = 64;
inline constexpr int kTileN = 64;
inline constexpr int kTileK = 64;
inline constexpr int kHeadDim = 64;
inline constexpr int kMaxEmbd = 2048;

// GEMM 的后处理: 加 bias; 加 bias 再做 GELU; 加 bias 再原地加到 y 上 (残差).
enum class Epilogue { kBias, kGelu, kResidual };

class KernelLibrary {
 public:
  // directory 中含 manifest.txt 和每个 kernel 的 cubin. 加载时检查 ABI 版本, 参数签名,
  // 线程数, 以及 cubin 中每个参数的字节数.
  explicit KernelLibrary(const std::filesystem::path& directory);
  KernelLibrary(const KernelLibrary&) = delete;
  KernelLibrary& operator=(const KernelLibrary&) = delete;

  const std::string& backend() const { return backend_; }
  const std::string& arch() const { return arch_; }

  // x[r, :] = wte[ids[r], :] + wpe[pos0 + r, :]
  void embedding(CUstream stream, CUdeviceptr x, CUdeviceptr ids, CUdeviceptr wte,
                 CUdeviceptr wpe, int rows, int pos0, int n_embd) const;
  // y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * gamma + beta
  void layernorm(CUstream stream, CUdeviceptr y, CUdeviceptr x, CUdeviceptr gamma,
                 CUdeviceptr beta, int rows, int n_embd, float eps) const;
  // y = epilogue(x @ w^T + b). x 为 (m, k), w 为 (n, k), b 为 (n,).
  void linear(CUstream stream, Epilogue epilogue, CUdeviceptr y, CUdeviceptr x, CUdeviceptr w,
              CUdeviceptr b, int m, int n, int k) const;
  // y = x @ w^T, 输出 FP32.
  void lm_head(CUstream stream, CUdeviceptr y, CUdeviceptr x, CUdeviceptr w, int m, int n,
               int k) const;
  // 因果注意力. 第 i 个 query 位于 pos0 + i, 看到缓存中位置 0 .. pos0 + i 的 key.
  // qkv 为 (m, 3 * n_head * 64); 缓存为 (位置, n_head * 64), 已含本次的 key 与 value.
  void attention(CUstream stream, CUdeviceptr y, CUdeviceptr qkv, CUdeviceptr k_cache,
                 CUdeviceptr v_cache, int m, int pos0, int n_head, float scale) const;

 private:
  // 析构时卸载 cubin 模块.
  struct ModuleDeleter {
    void operator()(CUmodule module) const { cuModuleUnload(module); }
  };
  struct Kernel {
    std::unique_ptr<CUmod_st, ModuleDeleter> module;
    CUfunction function = nullptr;
    unsigned smem_bytes = 0;
  };

  template <typename... Args>
  void launch(const char* name, CUstream stream, unsigned grid_x, unsigned grid_y,
              Args... args) const;

  std::string backend_;
  std::string arch_;
  std::map<std::string, Kernel, std::less<>> kernels_;
};

}  // namespace gpt2
