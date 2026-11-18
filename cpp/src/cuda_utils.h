// CUDA driver API 的辅助类型: 错误检查, context 与 stream, 设备内存.
//
// 本项目只用 driver API (libcuda), 不用 runtime API (libcudart). kernel 来自 cubin 文件,
// 由 cuModuleLoad 加载, 由 cuLaunchKernel 启动, 所以 C++ 程序不需要 nvcc 编译.
#pragma once

#include <cuda.h>

#include <cstddef>
#include <string>

namespace gpt2 {

// 检查 driver API 的返回值. 失败时抛出 std::runtime_error, 信息含表达式与错误名.
void check_cuda(CUresult result, const char* expr, const char* file, int line);

#define CU_CHECK(expr) ::gpt2::check_cuda((expr), #expr, __FILE__, __LINE__)

// 当前线程使用的 CUDA context (设备的 primary context) 和一个 stream.
// 其他持有设备资源的对象必须先于它析构.
class CudaContext {
 public:
  explicit CudaContext(int device_ordinal);
  ~CudaContext();
  CudaContext(const CudaContext&) = delete;
  CudaContext& operator=(const CudaContext&) = delete;

  CUstream stream() const { return stream_; }
  const std::string& device_name() const { return device_name_; }

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  CUstream stream_ = nullptr;
  std::string device_name_;
};

// 一块设备内存. 构造时清零, 析构时释放.
// 清零与 upload 在默认 stream 上排队; CudaContext 的 stream 会等待它们完成.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes);
  ~DeviceBuffer();
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  CUdeviceptr ptr() const { return ptr_; }
  size_t bytes() const { return bytes_; }

  // 从主机内存复制 bytes 字节. 同步执行.
  void upload(const void* host, size_t bytes);

 private:
  void release() noexcept;

  CUdeviceptr ptr_ = 0;
  size_t bytes_ = 0;
};

}  // namespace gpt2
