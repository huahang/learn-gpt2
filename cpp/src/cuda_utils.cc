#include "cuda_utils.h"

#include <format>
#include <stdexcept>
#include <utility>

namespace gpt2 {

void check_cuda(CUresult result, const char* expr, const char* file, int line) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  throw std::runtime_error(std::format("CUDA 错误 {}: {}\n  调用: {}\n  位置: {}:{}",
                                       name ? name : "未知", description ? description : "",
                                       expr, file, line));
}

CudaContext::CudaContext(int device_ordinal) {
  CU_CHECK(cuInit(0));
  int count = 0;
  CU_CHECK(cuDeviceGetCount(&count));
  if (device_ordinal < 0 || device_ordinal >= count) {
    throw std::runtime_error(std::format("没有编号为 {} 的 GPU, 共有 {} 个", device_ordinal, count));
  }
  CU_CHECK(cuDeviceGet(&device_, device_ordinal));
  CU_CHECK(cuDevicePrimaryCtxRetain(&context_, device_));
  CU_CHECK(cuCtxSetCurrent(context_));
  // 阻塞式 stream: 它与默认 stream (legacy NULL stream) 隐式同步. DeviceBuffer 的清零和
  // 同步复制在默认 stream 上执行, 可能在函数返回后才完成; 隐式同步保证 kernel 在它们之后运行.
  CU_CHECK(cuStreamCreate(&stream_, CU_STREAM_DEFAULT));
  char name[256] = {};
  CU_CHECK(cuDeviceGetName(name, sizeof(name), device_));
  device_name_ = name;
}

CudaContext::~CudaContext() {
  if (stream_ != nullptr) {
    cuStreamDestroy(stream_);
  }
  if (context_ != nullptr) {
    cuDevicePrimaryCtxRelease(device_);
  }
}

DeviceBuffer::DeviceBuffer(size_t bytes) : bytes_(bytes) {
  if (bytes == 0) {
    return;
  }
  CU_CHECK(cuMemAlloc(&ptr_, bytes));
  CU_CHECK(cuMemsetD8(ptr_, 0, bytes));
}

DeviceBuffer::~DeviceBuffer() { release(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, 0)), bytes_(std::exchange(other.bytes_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    release();
    ptr_ = std::exchange(other.ptr_, 0);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

void DeviceBuffer::upload(const void* host, size_t bytes) {
  if (bytes > bytes_) {
    throw std::runtime_error(std::format("复制 {} 字节超出设备内存大小 {}", bytes, bytes_));
  }
  CU_CHECK(cuMemcpyHtoD(ptr_, host, bytes));
}

void DeviceBuffer::release() noexcept {
  if (ptr_ != 0) {
    cuMemFree(ptr_);
    ptr_ = 0;
  }
  bytes_ = 0;
}

}  // namespace gpt2
