// UTF-8 编码与解码.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace gpt2 {

// 把码点 code 编码为 UTF-8, 追加到 out.
inline void append_utf8(std::string& out, uint32_t code) {
  if (code < 0x80) {
    out.push_back(static_cast<char>(code));
  } else if (code < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (code >> 6)));
    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else if (code < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (code >> 12)));
    out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (code >> 18)));
    out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  }
}

// 从 text[pos] 解码一个码点, 并把 pos 移到下一个字符. 无效的字节按单个字节处理,
// 返回 0xFFFD (替换字符).
inline uint32_t next_utf8(std::string_view text, size_t& pos) {
  const auto byte = [&](size_t i) { return static_cast<uint8_t>(text[i]); };
  const uint8_t lead = byte(pos);
  int length = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3
                               : (lead >> 3) == 0x1E ? 4 : 0;
  if (length == 0 || pos + length > text.size()) {
    ++pos;
    return lead < 0x80 ? lead : 0xFFFD;
  }
  uint32_t code = length == 1 ? lead : lead & (0x7F >> length);
  for (int i = 1; i < length; ++i) {
    if ((byte(pos + i) & 0xC0) != 0x80) {
      ++pos;
      return 0xFFFD;
    }
    code = (code << 6) | (byte(pos + i) & 0x3F);
  }
  pos += length;
  return code;
}

}  // namespace gpt2
