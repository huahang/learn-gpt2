// gpt2: 用 cubin kernel 做 GPT-2 推理的命令行程序.
//
// 示例 (MODEL 是 Hugging Face 检查点目录, 例如 ~/.cache/huggingface/hub/models--gpt2/snapshots/<hash>):
//   gpt2 --model MODEL --backend cute_dsl --prompt "Hello, I am a language model,"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_utils.h"
#include "kernels.h"
#include "model.h"
#include "tokenizer.h"

namespace fs = std::filesystem;
using gpt2::CudaContext;
using gpt2::Gpt2Model;
using gpt2::Gpt2Tokenizer;
using gpt2::KernelLibrary;
using gpt2::WeightLayout;

namespace {

constexpr const char* kUsage = R"(用法: gpt2 --model <路径> [选项]

  --model PATH           safetensors 文件, 或含 model.safetensors 的 Hugging Face 检查点目录
  --backend NAME         kernel 后端: tilelang (默认) 或 cute_dsl
  --kernels DIR          cubin 目录 (含 manifest.txt), 覆盖 --backend 选出的目录
  --prompt TEXT          提示词. 需要检查点目录中的 vocab.json 与 merges.txt
  --tokens IDS           用逗号分隔的 token 编号作为提示词, 不需要分词器
  --tokenizer DIR        vocab.json 与 merges.txt 所在目录, 默认是检查点所在目录
  --max-new-tokens N     生成的 token 数, 默认 32
  --temperature T        采样温度, 0 为贪心解码 (默认)
  --top-k K              只从概率最大的 K 个 token 中采样
  --seed S               随机数种子
  --device N             GPU 编号, 默认 0
  --weight-layout L      hf (Conv1D, 默认) 或 pytorch (nn.Linear)
  --print-tokens         输出 token 编号, 不输出文本
  --dump-logits FILE     计算提示词每个位置的 logits, 以 FP32 行主序写入 FILE, 然后退出
  -h, --help             显示本帮助
)";

struct Options {
  fs::path model;
  std::string backend = "tilelang";
  fs::path kernels;
  std::optional<std::string> prompt;
  std::optional<std::vector<int32_t>> tokens;
  fs::path tokenizer;
  int max_new_tokens = 32;
  float temperature = 0.0f;
  int top_k = 0;
  std::optional<uint64_t> seed;
  int device = 0;
  WeightLayout layout = WeightLayout::kConv1D;
  bool print_tokens = false;
  fs::path dump_logits;
};

template <typename T>
T parse_number(std::string_view text, std::string_view option) {
  T value{};
  auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size()) {
    throw std::runtime_error(std::format("{} 的值无效: {}", option, text));
  }
  return value;
}

std::vector<int32_t> parse_tokens(std::string_view text) {
  std::vector<int32_t> tokens;
  size_t begin = 0;
  while (begin < text.size()) {
    size_t end = text.find(',', begin);
    if (end == std::string_view::npos) {
      end = text.size();
    }
    tokens.push_back(parse_number<int32_t>(text.substr(begin, end - begin), "--tokens"));
    begin = end + 1;
  }
  return tokens;
}

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      std::cout << kUsage;
      std::exit(0);
    }
    if (arg == "--print-tokens") {
      opt.print_tokens = true;
      continue;
    }
    if (i + 1 >= argc) {
      throw std::runtime_error(std::format("未知选项或缺少值: {}\n\n{}", arg, kUsage));
    }
    const std::string_view value = argv[++i];
    if (arg == "--model") {
      opt.model = value;
    } else if (arg == "--backend") {
      opt.backend = value;
    } else if (arg == "--kernels") {
      opt.kernels = value;
    } else if (arg == "--prompt") {
      opt.prompt = std::string(value);
    } else if (arg == "--tokens") {
      opt.tokens = parse_tokens(value);
    } else if (arg == "--tokenizer") {
      opt.tokenizer = value;
    } else if (arg == "--max-new-tokens") {
      opt.max_new_tokens = parse_number<int>(value, arg);
    } else if (arg == "--temperature") {
      opt.temperature = parse_number<float>(value, arg);
    } else if (arg == "--top-k") {
      opt.top_k = parse_number<int>(value, arg);
    } else if (arg == "--seed") {
      opt.seed = parse_number<uint64_t>(value, arg);
    } else if (arg == "--device") {
      opt.device = parse_number<int>(value, arg);
    } else if (arg == "--weight-layout") {
      if (value != "hf" && value != "pytorch") {
        throw std::runtime_error("--weight-layout 只能是 hf 或 pytorch");
      }
      opt.layout = value == "hf" ? WeightLayout::kConv1D : WeightLayout::kLinear;
    } else if (arg == "--dump-logits") {
      opt.dump_logits = value;
    } else {
      throw std::runtime_error(std::format("未知选项: {}\n\n{}", arg, kUsage));
    }
  }
  if (opt.model.empty()) {
    throw std::runtime_error(std::format("需要 --model\n\n{}", kUsage));
  }
  if (opt.prompt && opt.tokens) {
    throw std::runtime_error("--prompt 与 --tokens 只能选一个");
  }
  if (opt.max_new_tokens < 0 || opt.top_k < 0 || !std::isfinite(opt.temperature) ||
      opt.temperature < 0) {
    throw std::runtime_error("--max-new-tokens, --top-k 与 --temperature 不能为负数");
  }
  return opt;
}

// 默认的 cubin 目录: 程序位于 <build>/bin/gpt2, cubin 位于 <build>/kernels/<后端>.
fs::path default_kernel_dir(const std::string& backend) {
  const fs::path exe = fs::read_symlink("/proc/self/exe");
  return exe.parent_path().parent_path() / "kernels" / backend;
}

int32_t sample(const std::vector<float>& logits, float temperature, int top_k,
               std::mt19937_64& rng) {
  if (temperature == 0.0f) {
    return static_cast<int32_t>(std::ranges::max_element(logits) - logits.begin());
  }
  std::vector<float> scaled(logits.size());
  std::ranges::transform(logits, scaled.begin(), [&](float x) { return x / temperature; });
  float threshold = -INFINITY;
  if (top_k > 0 && top_k < static_cast<int>(scaled.size())) {
    std::vector<float> sorted = scaled;
    std::ranges::nth_element(sorted, sorted.begin() + (top_k - 1), std::greater<>());
    threshold = sorted[top_k - 1];
  }
  const float max = *std::ranges::max_element(scaled);
  std::vector<double> weights(scaled.size());
  for (size_t i = 0; i < scaled.size(); ++i) {
    weights[i] = scaled[i] < threshold ? 0.0 : std::exp(double(scaled[i]) - max);
  }
  std::discrete_distribution<int32_t> distribution(weights.begin(), weights.end());
  return distribution(rng);
}

double milliseconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

int run(const Options& opt) {
  const auto load_start = std::chrono::steady_clock::now();
  CudaContext context(opt.device);
  const fs::path kernel_dir = opt.kernels.empty() ? default_kernel_dir(opt.backend) : opt.kernels;
  KernelLibrary kernels(kernel_dir);
  Gpt2Model model(opt.model, kernels, context.stream(), opt.layout);
  const gpt2::Gpt2Config& cfg = model.config();

  // 分词器是可选的: 用 --tokens 输入并用 --print-tokens 输出时不需要它.
  const fs::path tokenizer_dir = !opt.tokenizer.empty()     ? opt.tokenizer
                                 : fs::is_directory(opt.model) ? opt.model
                                                               : opt.model.parent_path();
  std::optional<Gpt2Tokenizer> tokenizer;
  if (fs::exists(tokenizer_dir / "vocab.json") && fs::exists(tokenizer_dir / "merges.txt")) {
    tokenizer.emplace(tokenizer_dir / "vocab.json", tokenizer_dir / "merges.txt");
  }
  std::vector<int32_t> prompt;
  if (opt.tokens) {
    prompt = *opt.tokens;
  } else {
    if (!tokenizer) {
      throw std::runtime_error(std::format(
          "{} 中没有 vocab.json 与 merges.txt. 请用 --tokenizer 指定目录, 或用 --tokens 输入",
          tokenizer_dir.string()));
    }
    prompt = tokenizer->encode(opt.prompt.value_or("Hello, I am a language model,"));
  }
  if (prompt.empty()) {
    prompt.push_back(cfg.vocab_size - 1);  // GPT-2 的 <|endoftext|> 是最后一个 token
  }
  std::cerr << std::format(
      "[gpt2] 设备 {}, kernel 后端 {} ({}), n_layer={} n_head={} n_embd={} n_ctx={}, "
      "加载用时 {:.0f} ms\n",
      context.device_name(), kernels.backend(), kernels.arch(), cfg.n_layer, cfg.n_head,
      cfg.n_embd, cfg.n_ctx, milliseconds_since(load_start));

  if (!opt.dump_logits.empty()) {
    if (static_cast<int>(prompt.size()) > cfg.n_ctx) {
      throw std::runtime_error("--dump-logits 的提示词不能超过上下文长度");
    }
    const std::vector<float> logits = model.forward(prompt, 0, /*all_positions=*/true);
    std::ofstream out(opt.dump_logits, std::ios::binary);
    out.write(reinterpret_cast<const char*>(logits.data()),
              static_cast<std::streamsize>(logits.size() * sizeof(float)));
    if (!out) {
      throw std::runtime_error("无法写入 " + opt.dump_logits.string());
    }
    std::cerr << std::format("[gpt2] 已写入 {} x {} 个 logits 到 {}\n", prompt.size(),
                             cfg.vocab_size, opt.dump_logits.string());
    return 0;
  }

  const bool print_text = tokenizer && !opt.print_tokens;
  if (print_text) {
    std::cout << tokenizer->decode(prompt) << std::flush;
  }
  std::mt19937_64 rng(opt.seed.value_or(std::random_device{}()));
  std::vector<int32_t> sequence = prompt;
  double prefill_ms = 0;
  double decode_ms = 0;
  for (int step = 0; step < opt.max_new_tokens; ++step) {
    const int end = static_cast<int>(sequence.size());
    const auto start = std::chrono::steady_clock::now();
    std::vector<float> logits;
    // 第一步处理整个提示词. 之后每步只处理最新的 token, 前面的 key 与 value 已在缓存中.
    // GPT-2 使用绝对位置嵌入: 序列超过上下文长度后, 每步重新计算最后 n_ctx 个 token,
    // 与 learn_gpt2.gpt2.GPT2.generate 的做法相同.
    if (step == 0 || end > cfg.n_ctx) {
      const int begin = std::max(0, end - cfg.n_ctx);
      logits = model.forward(std::span(sequence).subspan(begin), 0);
    } else {
      logits = model.forward(std::span(sequence).subspan(end - 1, 1), end - 1);
    }
    (step == 0 ? prefill_ms : decode_ms) += milliseconds_since(start);
    const int32_t next = sample(logits, opt.temperature, opt.top_k, rng);
    sequence.push_back(next);
    if (print_text) {
      std::cout << tokenizer->decode(std::span(&next, 1)) << std::flush;
    }
  }
  if (print_text) {
    std::cout << "\n";
  } else {
    for (size_t i = 0; i < sequence.size(); ++i) {
      std::cout << (i ? " " : "") << sequence[i];
    }
    std::cout << "\n";
  }
  if (opt.max_new_tokens > 0) {
    std::cerr << std::format("[gpt2] 提示词 {} 个 token, 预填充 {:.1f} ms", prompt.size(),
                             prefill_ms);
    if (opt.max_new_tokens > 1) {
      const double per_token = decode_ms / (opt.max_new_tokens - 1);
      std::cerr << std::format(", 解码 {:.2f} ms/token ({:.1f} token/s)", per_token,
                               1000.0 / per_token);
    }
    std::cerr << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_args(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "错误: " << error.what() << "\n";
    return 1;
  }
}
