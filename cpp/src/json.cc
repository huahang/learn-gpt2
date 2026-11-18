#include "json.h"

#include <charconv>
#include <cstdint>
#include <format>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "utf8.h"

namespace gpt2 {
namespace {

class Parser {
 public:
  explicit Parser(std::string_view text) : text_(text) {}

  JsonValue parse_document() {
    JsonValue value = parse_value();
    skip_space();
    if (pos_ != text_.size()) {
      fail("值之后有多余的内容");
    }
    return value;
  }

 private:
  [[noreturn]] void fail(std::string_view message) const {
    throw std::runtime_error(std::format("JSON 解析错误 (字节 {}): {}", pos_, message));
  }

  void skip_space() {
    while (pos_ < text_.size() &&
           (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' ||
            text_[pos_] == '\r')) {
      ++pos_;
    }
  }

  char peek() {
    skip_space();
    if (pos_ >= text_.size()) {
      fail("内容意外结束");
    }
    return text_[pos_];
  }

  void expect(char c) {
    if (peek() != c) {
      fail(std::format("预期 '{}'", c));
    }
    ++pos_;
  }

  bool consume_literal(std::string_view literal) {
    if (text_.substr(pos_, literal.size()) == literal) {
      pos_ += literal.size();
      return true;
    }
    return false;
  }

  JsonValue parse_value() {
    JsonValue value;
    const char c = peek();
    if (c == '{') {
      value.type = JsonValue::Type::kObject;
      ++pos_;
      if (peek() == '}') {
        ++pos_;
        return value;
      }
      while (true) {
        if (peek() != '"') {
          fail("对象的键必须是字符串");
        }
        std::string key = parse_string();
        expect(':');
        value.object.push_back({std::move(key), parse_value()});
        if (peek() == ',') {
          ++pos_;
          continue;
        }
        expect('}');
        return value;
      }
    }
    if (c == '[') {
      value.type = JsonValue::Type::kArray;
      ++pos_;
      if (peek() == ']') {
        ++pos_;
        return value;
      }
      while (true) {
        value.array.push_back(parse_value());
        if (peek() == ',') {
          ++pos_;
          continue;
        }
        expect(']');
        return value;
      }
    }
    if (c == '"') {
      value.type = JsonValue::Type::kString;
      value.string = parse_string();
      return value;
    }
    if (consume_literal("true")) {
      value.type = JsonValue::Type::kBool;
      value.boolean = true;
      return value;
    }
    if (consume_literal("false")) {
      value.type = JsonValue::Type::kBool;
      return value;
    }
    if (consume_literal("null")) {
      return value;
    }
    value.type = JsonValue::Type::kNumber;
    value.number = parse_number();
    return value;
  }

  double parse_number() {
    const char* begin = text_.data() + pos_;
    const char* end = text_.data() + text_.size();
    double number = 0;
    auto [next, error] = std::from_chars(begin, end, number);
    if (error != std::errc()) {
      fail("无法解析的值");
    }
    pos_ += static_cast<size_t>(next - begin);
    return number;
  }

  uint32_t parse_hex4() {
    if (pos_ + 4 > text_.size()) {
      fail("\\u 之后需要 4 位十六进制数");
    }
    uint32_t value = 0;
    auto [next, error] = std::from_chars(text_.data() + pos_, text_.data() + pos_ + 4, value, 16);
    if (error != std::errc() || next != text_.data() + pos_ + 4) {
      fail("\\u 之后需要 4 位十六进制数");
    }
    pos_ += 4;
    return value;
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) {
        fail("字符串没有结束");
      }
      const char c = text_[pos_++];
      if (c == '"') {
        return out;
      }
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (pos_ >= text_.size()) {
        fail("字符串没有结束");
      }
      const char escape = text_[pos_++];
      switch (escape) {
        case '"':
        case '\\':
        case '/':
          out.push_back(escape);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          uint32_t code = parse_hex4();
          // UTF-16 代理对: 高位 D800-DBFF 之后必须跟低位 DC00-DFFF.
          if (code >= 0xD800 && code <= 0xDBFF) {
            if (!consume_literal("\\u")) {
              fail("缺少代理对的低位");
            }
            const uint32_t low = parse_hex4();
            if (low < 0xDC00 || low > 0xDFFF) {
              fail("代理对的低位无效");
            }
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
          }
          append_utf8(out, code);
          break;
        }
        default:
          fail("未知的转义字符");
      }
    }
  }

  std::string_view text_;
  size_t pos_ = 0;
};

}  // namespace

const JsonValue* JsonValue::find(std::string_view key) const {
  for (const JsonMember& member : object) {
    if (member.key == key) {
      return &member.value;
    }
  }
  return nullptr;
}

JsonValue parse_json(std::string_view text) { return Parser(text).parse_document(); }

JsonValue read_json_file(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error(std::format("无法打开 {}", path.string()));
  }
  std::ostringstream content;
  content << file.rdbuf();
  try {
    return parse_json(content.str());
  } catch (const std::runtime_error& error) {
    throw std::runtime_error(std::format("{}: {}", path.string(), error.what()));
  }
}

}  // namespace gpt2
