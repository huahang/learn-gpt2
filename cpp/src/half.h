// FP32, BF16 与 FP16 之间的转换. 只用整数运算和 float 加法, 不依赖编译器的扩展类型.
#pragma once

#include <bit>
#include <cstdint>

namespace gpt2 {

// FP32 转为 FP16 的位模式, 舍入到最近的偶数, 与 PyTorch 的 tensor.half() 相同.
// 超出 FP16 范围的值变为 Inf, NaN 保持为 NaN.
inline uint16_t float_to_half(float value) {
  constexpr uint32_t kF32Inf = 255u << 23;
  constexpr uint32_t kF16Overflow = (127u + 16u) << 23;  // 2^16: 不小于它的值舍入为 Inf
  constexpr uint32_t kF16MinNormal = 113u << 23;         // 2^-14: FP16 最小的规格化数
  // 2^-1. 非规格化数加上它后, float 的尾数低 10 位正好是 FP16 的尾数, 舍入由 FPU 完成.
  constexpr uint32_t kDenormMagic = ((127u - 15u) + (23u - 10u) + 1u) << 23;

  uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t sign = bits & 0x80000000u;
  bits ^= sign;
  uint32_t half = 0;
  if (bits >= kF16Overflow) {
    half = bits > kF32Inf ? 0x7E00u : 0x7C00u;
  } else if (bits < kF16MinNormal) {
    const float sum = std::bit_cast<float>(bits) + std::bit_cast<float>(kDenormMagic);
    half = std::bit_cast<uint32_t>(sum) - kDenormMagic;
  } else {
    const uint32_t odd = (bits >> 13) & 1u;  // 结果尾数的最低位, 用于把 0.5 舍入到偶数
    bits += (uint32_t(15 - 127) << 23) + 0xFFFu;  // 调整指数偏置, 并加上不到一半的舍入量
    bits += odd;
    half = bits >> 13;
  }
  return static_cast<uint16_t>(half | (sign >> 16));
}

// BF16 是 FP32 的高 16 位, 转为 FP32 没有误差.
inline float bfloat16_to_float(uint16_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

}  // namespace gpt2
