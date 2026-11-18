# safetensors-cpp (vendored)

- 上游: https://github.com/syoyo/safetensors-cpp
- 版本: commit `af90b6c3006cdcecf8b7d7254f5f32d301728acc` (2025-12-28)
- 许可证: MIT, 见 [LICENSE](LICENSE)

本目录只保留 C++ 加载器需要的文件, 内容与上游一致, 没有修改:

- `safetensors.hh`: 头文件, 定义 `SAFETENSORS_CPP_IMPLEMENTATION` 时同时包含实现.
- `safetensors.cc`: 定义上述宏的编译单元.
- `LICENSE`, `README.md`: 上游的许可证与说明.

上游的 C API (`safetensors-c.*`, `csafetensors.*`), 示例, 测试, fuzzer 与 CMake 文件没有收录.
更新时从上游复制这几个文件, 并修改上面的 commit.
