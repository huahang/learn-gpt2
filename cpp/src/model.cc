#include "model.h"

#include <cmath>
#include <cstring>
#include <format>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include "half.h"
#include "json.h"
#include "safetensors.hh"

namespace gpt2 {
namespace {

using Half = uint16_t;  // FP16 的位模式

int round_up(int n, int multiple) { return (n + multiple - 1) / multiple * multiple; }

std::string shape_string(const std::vector<size_t>& shape) {
  std::string out = "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    out += std::format("{}{}", i ? ", " : "", shape[i]);
  }
  return out + ")";
}

// 一个 safetensors 文件. 用 mmap 读取, 不复制整个文件.
class Checkpoint {
 public:
  explicit Checkpoint(const std::filesystem::path& file) {
    std::string warn;
    std::string err;
    if (!safetensors::mmap_from_file(file.string(), &st_, &warn, &err)) {
      throw std::runtime_error(std::format("无法读取 {}: {}", file.string(), err));
    }
    // 逐个检查数据范围. 不用 safetensors::validate_data_offsets: 它假定文件头中最后一个
    // 张量的数据位于文件末尾, 而 Hugging Face 的 GPT-2 文件头按名字排序, 数据另有顺序.
    const uint8_t* base = st_.mmaped ? st_.databuffer_addr : st_.storage.data();
    const size_t data_size = st_.mmaped ? st_.databuffer_size : st_.storage.size();
    for (const std::string& key : st_.tensors.keys()) {
      safetensors::tensor_t tensor;
      st_.tensors.at(key, &tensor);
      const auto [begin, end] = tensor.data_offsets;
      const size_t bytes = safetensors::get_dtype_bytes(tensor.dtype) *
                           safetensors::get_shape_size(tensor);
      if (begin > end || end > data_size || end - begin != bytes) {
        throw std::runtime_error(std::format("{}: 张量 {} 的数据范围 [{}, {}) 无效",
                                             file.string(), key, begin, end));
      }
      // GPT2LMHeadModel 的键带 "transformer." 前缀, GPT2Model 的键不带.
      const std::string name = key.starts_with("transformer.") ? key.substr(12) : key;
      // 旧格式保存的因果掩码缓冲区, 以及与 wte 共享的 LM head, 不含额外信息.
      if (name == "lm_head.weight" || name.ends_with(".attn.bias") ||
          name.ends_with(".attn.masked_bias")) {
        continue;
      }
      if (!tensors_.emplace(name, Entry{tensor, base + tensor.data_offsets[0]}).second) {
        throw std::runtime_error(std::format("{} 中有重复的权重 {}", file.string(), name));
      }
    }
  }

  const std::vector<size_t>& shape(const std::string& name) const { return find(name).tensor.shape; }

  bool contains(const std::string& name) const { return tensors_.contains(name); }

  std::vector<std::string> names() const {
    std::vector<std::string> out;
    for (const auto& [name, entry] : tensors_) {
      out.push_back(name);
    }
    return out;
  }

  // 检查形状为 shape 后转为 FP16. transpose 为 true 时把 (rows, cols) 转置为 (cols, rows).
  std::vector<Half> to_half(const std::string& name, const std::vector<size_t>& shape,
                            bool transpose = false) {
    const Entry& entry = find(name);
    used_.insert(name);
    if (entry.tensor.shape != shape) {
      throw std::runtime_error(std::format("权重 {} 的形状为 {}, 预期 {}", name,
                                           shape_string(entry.tensor.shape), shape_string(shape)));
    }
    const size_t rows = shape[0];
    const size_t cols = shape.size() == 2 ? shape[1] : 1;
    std::vector<Half> out(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        out[transpose ? c * rows + r : r * cols + c] = load(entry, r * cols + c);
      }
    }
    return out;
  }

  // 没有被 to_half 读取的权重. 检查点不应含有多余的权重.
  std::vector<std::string> unused() const {
    std::vector<std::string> out;
    for (const auto& [name, entry] : tensors_) {
      if (!used_.contains(name)) {
        out.push_back(name);
      }
    }
    return out;
  }

 private:
  struct Entry {
    safetensors::tensor_t tensor;
    const uint8_t* data;
  };

  const Entry& find(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      throw std::runtime_error("检查点缺少权重 " + name);
    }
    return it->second;
  }

  // 读取第 i 个元素并转为 FP16. FP32 与 BF16 的转换舍入到最近的偶数, 与 PyTorch 相同.
  static Half load(const Entry& entry, size_t i) {
    uint16_t bits16 = 0;
    switch (entry.tensor.dtype) {
      case safetensors::kFLOAT16:
        std::memcpy(&bits16, entry.data + 2 * i, 2);
        return bits16;
      case safetensors::kBFLOAT16:
        std::memcpy(&bits16, entry.data + 2 * i, 2);
        return float_to_half(bfloat16_to_float(bits16));
      case safetensors::kFLOAT32: {
        float value = 0;
        std::memcpy(&value, entry.data + 4 * i, 4);
        return float_to_half(value);
      }
      default:
        throw std::runtime_error("权重只支持 F32, F16 与 BF16, 实际为 " +
                                 safetensors::get_dtype_str(entry.tensor.dtype));
    }
  }

  safetensors::safetensors_t st_;
  std::map<std::string, Entry> tensors_;
  std::set<std::string> used_;
};

DeviceBuffer upload(const std::vector<Half>& host) {
  DeviceBuffer buffer(host.size() * sizeof(Half));
  buffer.upload(host.data(), buffer.bytes());
  return buffer;
}

}  // namespace

Gpt2Model::Gpt2Model(const std::filesystem::path& checkpoint, const KernelLibrary& kernels,
                     CUstream stream, WeightLayout layout)
    : kernels_(kernels), stream_(stream) {
  const bool is_dir = std::filesystem::is_directory(checkpoint);
  const std::filesystem::path file = is_dir ? checkpoint / "model.safetensors" : checkpoint;
  const std::filesystem::path directory = is_dir ? checkpoint : checkpoint.parent_path();
  Checkpoint ckpt(file);

  // 形状推出词表, 隐藏维度, 上下文长度与层数.
  const std::vector<size_t>& wte_shape = ckpt.shape("wte.weight");
  const std::vector<size_t>& wpe_shape = ckpt.shape("wpe.weight");
  if (wte_shape.size() != 2 || wpe_shape.size() != 2) {
    throw std::runtime_error("wte.weight 与 wpe.weight 必须是二维张量");
  }
  Gpt2Config& cfg = config_;
  cfg.vocab_size = static_cast<int>(wte_shape[0]);
  cfg.n_embd = static_cast<int>(wte_shape[1]);
  cfg.n_ctx = static_cast<int>(wpe_shape[0]);
  while (ckpt.contains(std::format("h.{}.ln_1.weight", cfg.n_layer))) {
    ++cfg.n_layer;
  }
  cfg.n_head = cfg.n_embd / kHeadDim;
  const std::filesystem::path config_json = directory / "config.json";
  if (std::filesystem::exists(config_json)) {
    const JsonValue json = read_json_file(config_json);
    if (const JsonValue* v = json.find("n_head"); v != nullptr) {
      cfg.n_head = static_cast<int>(v->number);
    }
    if (const JsonValue* v = json.find("layer_norm_epsilon"); v != nullptr) {
      cfg.layer_norm_eps = static_cast<float>(v->number);
    }
  }
  cfg.padded_vocab = round_up(cfg.vocab_size, kTileN);

  // kernel 的前提条件, 见 kernel_abi.py.
  const int c = cfg.n_embd;
  if (cfg.n_layer == 0 || c % 64 != 0 || c > kMaxEmbd || cfg.n_head * kHeadDim != c) {
    throw std::runtime_error(std::format(
        "kernel 需要 n_embd 为 64 的倍数且不超过 {}, 每个 head 64 维. 检查点: n_layer={}, "
        "n_embd={}, n_head={}",
        kMaxEmbd, cfg.n_layer, c, cfg.n_head));
  }

  // 词嵌入补齐到 padded_vocab 行, 补齐的行为零, 对应的 logits 为 0, 返回前被截掉.
  std::vector<Half> wte = ckpt.to_half("wte.weight", wte_shape);
  wte.resize(static_cast<size_t>(cfg.padded_vocab) * c, 0);
  wte_ = upload(wte);
  wpe_ = upload(ckpt.to_half("wpe.weight", wpe_shape));
  ln_f_gamma_ = upload(ckpt.to_half("ln_f.weight", {size_t(c)}));
  ln_f_beta_ = upload(ckpt.to_half("ln_f.bias", {size_t(c)}));

  rows_ = round_up(cfg.n_ctx, kTileM);
  const size_t row_bytes = size_t(c) * sizeof(Half);
  // kernel 需要 (out, in) 布局的矩阵, 即 y = x @ w^T.
  const auto matrix_param = [&](const std::string& name, size_t out, size_t in) {
    if (layout == WeightLayout::kConv1D) {
      return upload(ckpt.to_half(name, {in, out}, /*transpose=*/true));
    }
    return upload(ckpt.to_half(name, {out, in}));
  };
  const auto vector_param = [&](const std::string& name, size_t size) {
    return upload(ckpt.to_half(name, {size}));
  };
  for (int i = 0; i < cfg.n_layer; ++i) {
    const std::string p = std::format("h.{}.", i);
    Layer layer;
    layer.ln_1_gamma = vector_param(p + "ln_1.weight", c);
    layer.ln_1_beta = vector_param(p + "ln_1.bias", c);
    layer.attn_w = matrix_param(p + "attn.c_attn.weight", 3 * c, c);
    layer.attn_b = vector_param(p + "attn.c_attn.bias", 3 * c);
    layer.proj_w = matrix_param(p + "attn.c_proj.weight", c, c);
    layer.proj_b = vector_param(p + "attn.c_proj.bias", c);
    layer.ln_2_gamma = vector_param(p + "ln_2.weight", c);
    layer.ln_2_beta = vector_param(p + "ln_2.bias", c);
    layer.fc_w = matrix_param(p + "mlp.c_fc.weight", 4 * c, c);
    layer.fc_b = vector_param(p + "mlp.c_fc.bias", 4 * c);
    layer.mlp_proj_w = matrix_param(p + "mlp.c_proj.weight", c, 4 * c);
    layer.mlp_proj_b = vector_param(p + "mlp.c_proj.bias", c);
    // 缓存清零: 注意力 kernel 会读取尚未写入的行 (再用掩码去掉), 这些行必须是有限值.
    layer.k_cache = DeviceBuffer(rows_ * row_bytes);
    layer.v_cache = DeviceBuffer(rows_ * row_bytes);
    layers_.push_back(std::move(layer));
  }
  if (const std::vector<std::string> extra = ckpt.unused(); !extra.empty()) {
    throw std::runtime_error(std::format("检查点中有 {} 个无法识别的权重, 例如 {}", extra.size(),
                                         extra.front()));
  }

  ids_ = DeviceBuffer(rows_ * sizeof(int32_t));
  x_ = DeviceBuffer(rows_ * row_bytes);
  norm_ = DeviceBuffer(rows_ * row_bytes);
  qkv_ = DeviceBuffer(3 * rows_ * row_bytes);
  attn_ = DeviceBuffer(rows_ * row_bytes);
  hidden_ = DeviceBuffer(4 * rows_ * row_bytes);
}

void Gpt2Model::append_to_cache(const DeviceBuffer& cache, int part, int pos0, int m) const {
  const size_t row_bytes = size_t(config_.n_embd) * sizeof(Half);
  CUDA_MEMCPY2D copy = {};
  copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.srcDevice = qkv_.ptr() + part * row_bytes;
  copy.srcPitch = 3 * row_bytes;
  copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = cache.ptr() + pos0 * row_bytes;
  copy.dstPitch = row_bytes;
  copy.WidthInBytes = row_bytes;
  copy.Height = static_cast<size_t>(m);
  CU_CHECK(cuMemcpy2DAsync(&copy, stream_));
}

std::vector<float> Gpt2Model::forward(std::span<const int32_t> tokens, int pos0,
                                      bool all_positions) {
  const Gpt2Config& cfg = config_;
  const int m = static_cast<int>(tokens.size());
  const int c = cfg.n_embd;
  if (m == 0 || pos0 < 0 || pos0 + m > cfg.n_ctx) {
    throw std::runtime_error(std::format("位置 {} 到 {} 超出上下文长度 {}", pos0, pos0 + m,
                                         cfg.n_ctx));
  }
  for (int32_t token : tokens) {
    if (token < 0 || token >= cfg.vocab_size) {
      throw std::runtime_error(std::format("token 编号 {} 超出词表大小 {}", token, cfg.vocab_size));
    }
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  const float eps = cfg.layer_norm_eps;

  CU_CHECK(cuMemcpyHtoDAsync(ids_.ptr(), tokens.data(), tokens.size_bytes(), stream_));
  kernels_.embedding(stream_, x_.ptr(), ids_.ptr(), wte_.ptr(), wpe_.ptr(), m, pos0, c);
  for (const Layer& layer : layers_) {
    // x = x + attn(ln_1(x))
    kernels_.layernorm(stream_, norm_.ptr(), x_.ptr(), layer.ln_1_gamma.ptr(),
                       layer.ln_1_beta.ptr(), m, c, eps);
    kernels_.linear(stream_, Epilogue::kBias, qkv_.ptr(), norm_.ptr(), layer.attn_w.ptr(),
                    layer.attn_b.ptr(), m, 3 * c, c);
    append_to_cache(layer.k_cache, 1, pos0, m);
    append_to_cache(layer.v_cache, 2, pos0, m);
    kernels_.attention(stream_, attn_.ptr(), qkv_.ptr(), layer.k_cache.ptr(),
                       layer.v_cache.ptr(), m, pos0, cfg.n_head, scale);
    kernels_.linear(stream_, Epilogue::kResidual, x_.ptr(), attn_.ptr(), layer.proj_w.ptr(),
                    layer.proj_b.ptr(), m, c, c);
    // x = x + mlp(ln_2(x))
    kernels_.layernorm(stream_, norm_.ptr(), x_.ptr(), layer.ln_2_gamma.ptr(),
                       layer.ln_2_beta.ptr(), m, c, eps);
    kernels_.linear(stream_, Epilogue::kGelu, hidden_.ptr(), norm_.ptr(), layer.fc_w.ptr(),
                    layer.fc_b.ptr(), m, 4 * c, c);
    kernels_.linear(stream_, Epilogue::kResidual, x_.ptr(), hidden_.ptr(),
                    layer.mlp_proj_w.ptr(), layer.mlp_proj_b.ptr(), m, c, 4 * c);
  }

  // 生成时只需要最后一个位置. 把它的 LayerNorm 结果写到 norm_ 的第 0 行, 这样 LM head
  // 从缓冲区起点读取, 补齐的行仍在缓冲区内.
  const int rows = all_positions ? m : 1;
  const size_t row_bytes = size_t(c) * sizeof(Half);
  const CUdeviceptr first = x_.ptr() + (all_positions ? 0 : size_t(m - 1) * row_bytes);
  kernels_.layernorm(stream_, norm_.ptr(), first, ln_f_gamma_.ptr(), ln_f_beta_.ptr(), rows, c,
                     eps);
  const size_t logits_bytes = size_t(rows) * cfg.padded_vocab * sizeof(float);
  if (logits_.bytes() < logits_bytes) {
    logits_ = DeviceBuffer(logits_bytes);
  }
  kernels_.lm_head(stream_, logits_.ptr(), norm_.ptr(), wte_.ptr(), rows, cfg.padded_vocab, c);

  std::vector<float> padded(size_t(rows) * cfg.padded_vocab);
  CU_CHECK(cuMemcpyDtoHAsync(padded.data(), logits_.ptr(), logits_bytes, stream_));
  CU_CHECK(cuStreamSynchronize(stream_));
  // 去掉补齐的词表列.
  std::vector<float> logits(size_t(rows) * cfg.vocab_size);
  for (int r = 0; r < rows; ++r) {
    std::memcpy(logits.data() + size_t(r) * cfg.vocab_size,
                padded.data() + size_t(r) * cfg.padded_vocab, cfg.vocab_size * sizeof(float));
  }
  return logits;
}

}  // namespace gpt2
