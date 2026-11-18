// GPT-2 模型: 从 safetensors 加载权重, 用 KernelLibrary 中的 cubin kernel 计算前向.
//
// 权重与激活都是 FP16, 累加与归一化在 kernel 内部使用 FP32. 每层的 KV 缓存按
// (位置, n_embd) 存放, head h 占第 h * 64 到 (h + 1) * 64 列, 与 QKV GEMM 的输出布局相同.
#pragma once

#include <cuda.h>

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "cuda_utils.h"
#include "kernels.h"

namespace gpt2 {

struct Gpt2Config {
  int n_layer = 0;
  int n_head = 0;
  int n_embd = 0;
  int n_ctx = 0;
  int vocab_size = 0;
  // 词表补齐到 kTileN 的倍数, 作为 LM head 的输出列数. 补齐的行为零.
  int padded_vocab = 0;
  float layer_norm_eps = 1e-5f;
};

// 矩阵权重的存储方式. Hugging Face 的 GPT-2 使用 Conv1D, 形状为 (输入, 输出);
// PyTorch 的 nn.Linear 形状为 (输出, 输入). c_proj 是方阵, 无法从形状区分, 必须指定.
enum class WeightLayout { kConv1D, kLinear };

class Gpt2Model {
 public:
  // checkpoint 是 safetensors 文件, 或含 model.safetensors 的目录. 同目录的 config.json
  // 可提供 n_head 与 layer_norm_epsilon; 没有时 n_head = n_embd / 64.
  Gpt2Model(const std::filesystem::path& checkpoint, const KernelLibrary& kernels,
            CUstream stream, WeightLayout layout);

  const Gpt2Config& config() const { return config_; }

  // 计算 tokens 的 logits. tokens 位于位置 pos0 .. pos0 + n - 1, 它们的 key 与 value
  // 写入缓存; 位置 0 .. pos0 - 1 必须已在缓存中. 返回最后一个位置的 vocab_size 个 logits;
  // all_positions 为 true 时返回每个位置的 logits, 共 n 行.
  std::vector<float> forward(std::span<const int32_t> tokens, int pos0,
                             bool all_positions = false);

 private:
  struct Layer {
    DeviceBuffer ln_1_gamma, ln_1_beta;
    DeviceBuffer attn_w, attn_b;  // c_attn: (3 * n_embd, n_embd)
    DeviceBuffer proj_w, proj_b;  // attn.c_proj: (n_embd, n_embd)
    DeviceBuffer ln_2_gamma, ln_2_beta;
    DeviceBuffer fc_w, fc_b;            // mlp.c_fc: (4 * n_embd, n_embd)
    DeviceBuffer mlp_proj_w, mlp_proj_b;  // mlp.c_proj: (n_embd, 4 * n_embd)
    DeviceBuffer k_cache, v_cache;      // (缓存行数, n_embd)
  };

  // 把本次 token 的 key 或 value (qkv 的第 part 部分) 复制到缓存的 pos0 .. pos0 + m - 1 行.
  void append_to_cache(const DeviceBuffer& cache, int part, int pos0, int m) const;

  const KernelLibrary& kernels_;
  CUstream stream_;
  Gpt2Config config_;

  DeviceBuffer wte_;  // (padded_vocab, n_embd), 同时作为 LM head 的权重
  DeviceBuffer wpe_;  // (n_ctx, n_embd)
  DeviceBuffer ln_f_gamma_, ln_f_beta_;
  std::vector<Layer> layers_;

  // 工作区. 行数补齐到 kTileM 的倍数, 满足 ABI 对补齐行的要求.
  int rows_ = 0;
  DeviceBuffer ids_;     // (rows,) int32
  DeviceBuffer x_;       // 残差流 (rows, n_embd)
  DeviceBuffer norm_;    // LayerNorm 的输出 (rows, n_embd)
  DeviceBuffer qkv_;     // (rows, 3 * n_embd)
  DeviceBuffer attn_;    // 注意力输出 (rows, n_embd)
  DeviceBuffer hidden_;  // MLP 中间层 (rows, 4 * n_embd)
  DeviceBuffer logits_;  // (行数, padded_vocab) FP32, 按需增长
};

}  // namespace gpt2
