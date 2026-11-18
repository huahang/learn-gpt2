#include "kernels.h"

#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "cuda_utils.h"

namespace gpt2 {
namespace {

// C++ 端预期的 kernel 与参数签名. 与 kernel_abi.py 中 Kernel.signature 逐字一致,
// 加载时与 manifest.txt 比较, 防止两边的接口不同步.
struct AbiKernel {
  const char* name;
  const char* signature;
};

constexpr AbiKernel kAbiKernels[] = {
    {"gpt2_embedding", "f16*,const_i32*,const_f16*,const_f16*,i32,i32,i32"},
    {"gpt2_layernorm", "f16*,const_f16*,const_f16*,const_f16*,i32,i32,f32"},
    {"gpt2_linear_bias", "f16*,const_f16*,const_f16*,const_f16*,i32,i32,i32"},
    {"gpt2_linear_bias_gelu", "f16*,const_f16*,const_f16*,const_f16*,i32,i32,i32"},
    {"gpt2_linear_bias_residual", "f16*,const_f16*,const_f16*,const_f16*,i32,i32,i32"},
    {"gpt2_lm_head", "f32*,const_f16*,const_f16*,i32,i32,i32"},
    {"gpt2_attention", "f16*,const_f16*,const_f16*,const_f16*,i32,i32,i32,f32"},
};

// 动态共享内存超过这个值时, 需要先用 cuFuncSetAttribute 提高上限.
constexpr unsigned kDefaultSmemLimit = 48 * 1024;

struct ManifestEntry {
  std::string symbol;
  int threads = 0;
  unsigned smem_bytes = 0;
  std::string signature;
};

// 签名中每个参数的字节数: 指针 8 字节, 标量 4 字节.
std::vector<size_t> param_sizes(std::string_view signature) {
  std::vector<size_t> sizes;
  size_t begin = 0;
  while (begin <= signature.size()) {
    size_t end = signature.find(',', begin);
    if (end == std::string_view::npos) {
      end = signature.size();
    }
    sizes.push_back(signature.substr(begin, end - begin).ends_with('*') ? 8 : 4);
    begin = end + 1;
  }
  return sizes;
}

// 用 driver API 读出 cubin 中每个参数的大小, 与签名比较. 这一步检查 cubin 本身,
// 而不只是 manifest 的描述.
void check_params(CUfunction function, const char* name, std::string_view signature) {
  const std::vector<size_t> expected = param_sizes(signature);
  for (size_t i = 0; i < expected.size(); ++i) {
    size_t offset = 0;
    size_t size = 0;
    if (cuFuncGetParamInfo(function, i, &offset, &size) != CUDA_SUCCESS) {
      throw std::runtime_error(std::format("{}: cubin 只有 {} 个参数, ABI 规定 {} 个", name, i,
                                           expected.size()));
    }
    if (size != expected[i]) {
      throw std::runtime_error(std::format("{}: 第 {} 个参数为 {} 字节, ABI 规定 {} 字节", name,
                                           i, size, expected[i]));
    }
  }
  size_t offset = 0;
  size_t size = 0;
  if (cuFuncGetParamInfo(function, expected.size(), &offset, &size) == CUDA_SUCCESS) {
    throw std::runtime_error(
        std::format("{}: cubin 的参数多于 ABI 规定的 {} 个", name, expected.size()));
  }
}

unsigned ceil_div(int a, int b) { return static_cast<unsigned>((a + b - 1) / b); }

}  // namespace

KernelLibrary::KernelLibrary(const std::filesystem::path& directory) {
  const std::filesystem::path manifest_path = directory / "manifest.txt";
  std::ifstream manifest(manifest_path);
  if (!manifest) {
    throw std::runtime_error(
        std::format("找不到 {}. 请先构建 kernel: make -C cpp kernels", manifest_path.string()));
  }

  int abi_version = 0;
  std::map<std::string, ManifestEntry, std::less<>> entries;
  std::string line;
  while (std::getline(manifest, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::istringstream fields(line);
    std::string key;
    fields >> key;
    if (key == "abi") {
      fields >> abi_version;
    } else if (key == "backend") {
      fields >> backend_;
    } else if (key == "arch") {
      fields >> arch_;
    } else if (key == "kernel") {
      std::string name;
      ManifestEntry entry;
      fields >> name >> entry.symbol >> entry.threads >> entry.smem_bytes >> entry.signature;
      if (!fields) {
        throw std::runtime_error(std::format("{}: 无法解析 \"{}\"", manifest_path.string(), line));
      }
      entries[name] = entry;
    }
  }
  if (abi_version != kAbiVersion) {
    throw std::runtime_error(std::format("{}: ABI 版本为 {}, 本程序需要 {}",
                                         manifest_path.string(), abi_version, kAbiVersion));
  }

  for (const AbiKernel& abi : kAbiKernels) {
    auto it = entries.find(abi.name);
    if (it == entries.end()) {
      throw std::runtime_error(std::format("{}: 缺少 kernel {}", manifest_path.string(), abi.name));
    }
    const ManifestEntry& entry = it->second;
    if (entry.signature != abi.signature || entry.threads != kThreads) {
      throw std::runtime_error(std::format("{}: kernel {} 的接口与 ABI 不一致",
                                           manifest_path.string(), abi.name));
    }

    const std::filesystem::path cubin = directory / (std::string(abi.name) + ".cubin");
    CUmodule module = nullptr;
    CUresult loaded = cuModuleLoad(&module, cubin.string().c_str());
    if (loaded == CUDA_ERROR_NO_BINARY_FOR_GPU) {
      throw std::runtime_error(std::format(
          "{} 为 {} 编译, 不能在当前 GPU 上运行. 请用 ARCH=sm_XX 重新构建 kernel",
          cubin.string(), arch_));
    }
    CU_CHECK(loaded);
    Kernel& kernel = kernels_[abi.name];
    kernel.module.reset(module);
    kernel.smem_bytes = entry.smem_bytes;
    CU_CHECK(cuModuleGetFunction(&kernel.function, module, entry.symbol.c_str()));
    check_params(kernel.function, abi.name, abi.signature);
    if (kernel.smem_bytes > kDefaultSmemLimit) {
      CU_CHECK(cuFuncSetAttribute(kernel.function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  static_cast<int>(kernel.smem_bytes)));
    }
  }
}

template <typename... Args>
void KernelLibrary::launch(const char* name, CUstream stream, unsigned grid_x, unsigned grid_y,
                           Args... args) const {
  const Kernel& kernel = kernels_.find(name)->second;
  // cuLaunchKernel 需要每个参数的地址. 参数的类型已由调用者固定为 ABI 规定的类型:
  // CUdeviceptr (8 字节), int 或 float (4 字节).
  void* params[] = {static_cast<void*>(&args)...};
  CU_CHECK(cuLaunchKernel(kernel.function, grid_x, grid_y, 1, kThreads, 1, 1, kernel.smem_bytes,
                          stream, params, nullptr));
}

void KernelLibrary::embedding(CUstream stream, CUdeviceptr x, CUdeviceptr ids, CUdeviceptr wte,
                              CUdeviceptr wpe, int rows, int pos0, int n_embd) const {
  launch("gpt2_embedding", stream, rows, 1, x, ids, wte, wpe, rows, pos0, n_embd);
}

void KernelLibrary::layernorm(CUstream stream, CUdeviceptr y, CUdeviceptr x, CUdeviceptr gamma,
                              CUdeviceptr beta, int rows, int n_embd, float eps) const {
  launch("gpt2_layernorm", stream, rows, 1, y, x, gamma, beta, rows, n_embd, eps);
}

void KernelLibrary::linear(CUstream stream, Epilogue epilogue, CUdeviceptr y, CUdeviceptr x,
                           CUdeviceptr w, CUdeviceptr b, int m, int n, int k) const {
  const char* name = epilogue == Epilogue::kBias   ? "gpt2_linear_bias"
                     : epilogue == Epilogue::kGelu ? "gpt2_linear_bias_gelu"
                                                   : "gpt2_linear_bias_residual";
  launch(name, stream, ceil_div(m, kTileM), n / kTileN, y, x, w, b, m, n, k);
}

void KernelLibrary::lm_head(CUstream stream, CUdeviceptr y, CUdeviceptr x, CUdeviceptr w, int m,
                            int n, int k) const {
  launch("gpt2_lm_head", stream, ceil_div(m, kTileM), n / kTileN, y, x, w, m, n, k);
}

void KernelLibrary::attention(CUstream stream, CUdeviceptr y, CUdeviceptr qkv,
                              CUdeviceptr k_cache, CUdeviceptr v_cache, int m, int pos0,
                              int n_head, float scale) const {
  launch("gpt2_attention", stream, ceil_div(m, kTileM), n_head, y, qkv, k_cache, v_cache, m, pos0,
         n_head, scale);
}

}  // namespace gpt2
