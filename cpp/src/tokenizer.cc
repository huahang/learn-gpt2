#include "tokenizer.h"

#include <climits>
#include <format>
#include <fstream>
#include <stdexcept>

#include "json.h"
#include "utf8.h"

namespace gpt2 {
namespace {

enum class CharClass { kSpace, kLetter, kNumber, kOther };

// Unicode White_Space.
bool is_space(uint32_t c) {
  return c == ' ' || (c >= 0x09 && c <= 0x0D) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
         c == 0x205F || c == 0x3000;
}

// \p{N} 的近似: 常见的数字字符区段.
bool is_number(uint32_t c) {
  return (c >= '0' && c <= '9') || c == 0xB2 || c == 0xB3 || c == 0xB9 ||
         (c >= 0xBC && c <= 0xBE) || (c >= 0x0660 && c <= 0x0669) ||
         (c >= 0x06F0 && c <= 0x06F9) || (c >= 0x0966 && c <= 0x096F) || c == 0x2070 ||
         (c >= 0x2074 && c <= 0x2079) || (c >= 0x2080 && c <= 0x2089) ||
         (c >= 0x2150 && c <= 0x2182) || (c >= 0x2185 && c <= 0x2189) ||
         (c >= 0x2460 && c <= 0x249B) || (c >= 0x24EA && c <= 0x24FF) ||
         (c >= 0x2776 && c <= 0x2793) || c == 0x3007 || (c >= 0x3021 && c <= 0x3029) ||
         (c >= 0xFF10 && c <= 0xFF19);
}

// 非 ASCII 的标点, 符号, 组合符号与格式字符的近似. 其他非 ASCII 字符视为字母.
bool is_other_non_ascii(uint32_t c) {
  return (c >= 0x80 && c <= 0xBF && c != 0xAA && c != 0xB5 && c != 0xBA) || c == 0xD7 ||
         c == 0xF7 || (c >= 0x0300 && c <= 0x036F) || c == 0x037E || c == 0x0387 ||
         (c >= 0x200B && c <= 0x200F) || (c >= 0x2010 && c <= 0x205E) ||
         (c >= 0x2060 && c <= 0x206F) || (c >= 0x207A && c <= 0x207E) ||
         (c >= 0x208A && c <= 0x208E) || (c >= 0x20A0 && c <= 0x214F) ||
         (c >= 0x2190 && c <= 0x2BFF) || (c >= 0x2E00 && c <= 0x2E7F) ||
         (c >= 0x3001 && c <= 0x3003) || (c >= 0x3008 && c <= 0x3020) || c == 0x3030 ||
         c == 0x30FB || (c >= 0xFE00 && c <= 0xFE0F) || (c >= 0xFE10 && c <= 0xFE19) ||
         (c >= 0xFE30 && c <= 0xFE6B) || c == 0xFEFF || (c >= 0xFF01 && c <= 0xFF0F) ||
         (c >= 0xFF1A && c <= 0xFF20) || (c >= 0xFF3B && c <= 0xFF40) ||
         (c >= 0xFF5B && c <= 0xFF65) || (c >= 0xFFE0 && c <= 0xFFEE) || c == 0xFFFD ||
         (c >= 0x1F000 && c <= 0x1FAFF);
}

CharClass classify(uint32_t c) {
  if (is_space(c)) {
    return CharClass::kSpace;
  }
  if (is_number(c)) {
    return CharClass::kNumber;
  }
  if (c < 0x80) {
    const bool letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    return letter ? CharClass::kLetter : CharClass::kOther;
  }
  return is_other_non_ascii(c) ? CharClass::kOther : CharClass::kLetter;
}

struct Char {
  uint32_t code;
  size_t begin;  // 在原文中的字节范围 [begin, end)
  size_t end;
  CharClass cls;
};

// 按 GPT-2 的正则表达式切分文本, 返回每个片段在原文中的字节范围.
std::vector<std::string_view> pretokenize(std::string_view text) {
  std::vector<Char> chars;
  for (size_t pos = 0; pos < text.size();) {
    const size_t begin = pos;
    const uint32_t code = next_utf8(text, pos);
    chars.push_back({code, begin, pos, classify(code)});
  }
  const size_t n = chars.size();
  std::vector<std::string_view> pieces;
  const auto emit = [&](size_t from, size_t to) {
    pieces.push_back(text.substr(chars[from].begin, chars[to - 1].end - chars[from].begin));
  };
  const auto is = [&](size_t i, char c) { return i < n && chars[i].code == uint32_t(c); };

  size_t i = 0;
  while (i < n) {
    // 's|'t|'re|'ve|'m|'ll|'d
    if (is(i, '\'')) {
      size_t length = 0;
      if (is(i + 1, 's') || is(i + 1, 't') || is(i + 1, 'm') || is(i + 1, 'd')) {
        length = 2;
      } else if ((is(i + 1, 'r') && is(i + 2, 'e')) || (is(i + 1, 'v') && is(i + 2, 'e')) ||
                 (is(i + 1, 'l') && is(i + 2, 'l'))) {
        length = 3;
      }
      if (length > 0) {
        emit(i, i + length);
        i += length;
        continue;
      }
    }
    // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`: 可带一个前导空格的同类字符串.
    bool matched = false;
    for (CharClass cls : {CharClass::kLetter, CharClass::kNumber, CharClass::kOther}) {
      size_t start = i;
      if (is(i, ' ') && i + 1 < n && chars[i + 1].cls == cls) {
        start = i + 1;
      } else if (chars[i].cls != cls) {
        continue;
      }
      size_t end = start;
      while (end < n && chars[end].cls == cls) {
        ++end;
      }
      emit(i, end);
      i = end;
      matched = true;
      break;
    }
    if (matched) {
      continue;
    }
    // `\s+(?!\S)|\s+`: 空白串之后还有文字时, 留下最后一个空白, 让它与后面的词合在一起.
    size_t end = i;
    while (end < n && chars[end].cls == CharClass::kSpace) {
      ++end;
    }
    if (end < n && end - i > 1) {
      --end;
    }
    emit(i, end);
    i = end;
  }
  return pieces;
}

}  // namespace

Gpt2Tokenizer::Gpt2Tokenizer(const std::filesystem::path& vocab_json,
                             const std::filesystem::path& merges_txt) {
  // bytes_to_unicode: 可见的 Latin-1 字节映射为自身, 其余字节依次映射为 U+0100 起的字符.
  int next = 0;
  for (int b = 0; b < 256; ++b) {
    const bool visible = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE;
    const uint32_t code = visible ? b : 256 + next++;
    append_utf8(byte_encoder_[b], code);
    byte_decoder_[code] = static_cast<uint8_t>(b);
  }

  const JsonValue vocab = read_json_file(vocab_json);
  if (vocab.type != JsonValue::Type::kObject) {
    throw std::runtime_error(std::format("{} 不是 JSON 对象", vocab_json.string()));
  }
  for (const auto& [token, id] : vocab.object) {
    if (id.type != JsonValue::Type::kNumber || id.number < 0) {
      throw std::runtime_error(std::format("{}: {} 的编号无效", vocab_json.string(), token));
    }
    const auto index = static_cast<size_t>(id.number);
    if (index >= decoder_.size()) {
      decoder_.resize(index + 1);
    }
    decoder_[index] = token;
    encoder_[token] = static_cast<int32_t>(index);
  }

  std::ifstream merges(merges_txt);
  if (!merges) {
    throw std::runtime_error(std::format("无法打开 {}", merges_txt.string()));
  }
  std::string line;
  int rank = 0;
  while (std::getline(merges, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty() || line.starts_with("#version")) {
      continue;
    }
    merge_ranks_.emplace(line, rank++);
  }
}

void Gpt2Tokenizer::bpe(std::string_view piece, std::vector<int32_t>& ids) const {
  const std::string key(piece);
  if (auto cached = cache_.find(key); cached != cache_.end()) {
    ids.insert(ids.end(), cached->second.begin(), cached->second.end());
    return;
  }
  std::vector<std::string> symbols;
  for (unsigned char byte : piece) {
    symbols.push_back(byte_encoder_[byte]);
  }
  // 每轮找出优先级最高的相邻符号对, 把它的所有出现都合并.
  while (symbols.size() > 1) {
    int best_rank = INT_MAX;
    size_t best = 0;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      auto it = merge_ranks_.find(symbols[i] + ' ' + symbols[i + 1]);
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best = i;
      }
    }
    if (best_rank == INT_MAX) {
      break;
    }
    const std::string left = symbols[best];
    const std::string right = symbols[best + 1];
    std::vector<std::string> merged;
    for (size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right) {
        merged.push_back(left + right);
        i += 2;
      } else {
        merged.push_back(symbols[i]);
        i += 1;
      }
    }
    symbols = std::move(merged);
  }
  std::vector<int32_t> result;
  for (const std::string& symbol : symbols) {
    auto it = encoder_.find(symbol);
    if (it == encoder_.end()) {
      throw std::runtime_error("词表中没有 BPE 符号 " + symbol);
    }
    result.push_back(it->second);
  }
  ids.insert(ids.end(), result.begin(), result.end());
  cache_.emplace(key, std::move(result));
}

std::vector<int32_t> Gpt2Tokenizer::encode(std::string_view text) const {
  std::vector<int32_t> ids;
  for (std::string_view piece : pretokenize(text)) {
    bpe(piece, ids);
  }
  return ids;
}

std::string Gpt2Tokenizer::decode(std::span<const int32_t> ids) const {
  std::string bytes;
  for (int32_t id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= decoder_.size()) {
      throw std::runtime_error(std::format("token 编号 {} 超出词表", id));
    }
    const std::string& token = decoder_[id];
    for (size_t pos = 0; pos < token.size();) {
      const uint32_t code = next_utf8(token, pos);
      auto it = byte_decoder_.find(code);
      if (it == byte_decoder_.end()) {
        throw std::runtime_error(std::format("token {} 含有无法映射为字节的字符", id));
      }
      bytes.push_back(static_cast<char>(it->second));
    }
  }
  return bytes;
}

}  // namespace gpt2
