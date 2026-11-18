// 读取 config.json 与 vocab.json 的最小 JSON 解析器.
//
// 支持完整的 JSON 语法, 包括 \uXXXX 转义与代理对 (surrogate pair). 数字统一保存为 double.
#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace gpt2 {

struct JsonMember;

struct JsonValue {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Type type = Type::kNull;
  bool boolean = false;
  double number = 0;
  std::string string;  // UTF-8
  std::vector<JsonValue> array;
  // 按出现顺序保存的键值对.
  std::vector<JsonMember> object;

  // 对象中键为 key 的值. 不是对象或没有该键时返回 nullptr.
  const JsonValue* find(std::string_view key) const;
};

struct JsonMember {
  std::string key;
  JsonValue value;
};

// 解析失败时抛出 std::runtime_error, 信息含出错的字节位置.
JsonValue parse_json(std::string_view text);
JsonValue read_json_file(const std::filesystem::path& path);

}  // namespace gpt2
