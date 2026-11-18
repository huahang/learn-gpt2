// GPT-2 的字节级 BPE 分词器, 读取 Hugging Face 检查点中的 vocab.json 与 merges.txt.
//
// 编码分三步:
// 1. 按 GPT-2 的正则表达式把文本切成片段:
//    's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// 2. 把片段的每个字节映射为一个可见的 Unicode 字符 (bytes_to_unicode).
// 3. 按 merges.txt 的优先级反复合并相邻的符号, 最后查 vocab.json 得到 token 编号.
//
// 第 1 步需要 Unicode 字符类别. 本实现不带 Unicode 数据表: ASCII 字符的分类是精确的;
// 其他字符按常见的空白, 数字与标点区段近似分类, 其余都视为字母.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gpt2 {

class Gpt2Tokenizer {
 public:
  Gpt2Tokenizer(const std::filesystem::path& vocab_json, const std::filesystem::path& merges_txt);

  std::vector<int32_t> encode(std::string_view text) const;
  // 返回 token 对应的原始字节. 一个多字节字符可能被拆到相邻的 token 中.
  std::string decode(std::span<const int32_t> ids) const;
  int vocab_size() const { return static_cast<int>(decoder_.size()); }

 private:
  void bpe(std::string_view piece, std::vector<int32_t>& ids) const;

  std::unordered_map<std::string, int32_t> encoder_;
  std::vector<std::string> decoder_;
  // 键为 "<左符号> <右符号>", 值为合并的优先级 (越小越先合并).
  std::unordered_map<std::string, int> merge_ranks_;
  // 字节 -> 映射后字符的 UTF-8 编码; 映射后字符的码点 -> 字节.
  std::array<std::string, 256> byte_encoder_;
  std::unordered_map<uint32_t, uint8_t> byte_decoder_;
  // 片段 -> token 编号. 文本中重复的词只做一次 BPE.
  mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
};

}  // namespace gpt2
