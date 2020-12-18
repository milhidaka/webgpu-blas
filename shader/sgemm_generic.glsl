#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer arrayA {
    float numbers[];
} array_a;

layout(std430, set = 0, binding = 1) readonly buffer arrayB {
    float numbers[];
} array_b;

layout(std430, set = 0, binding = 2) buffer arrayC {
    float numbers[];
} array_c;

layout(std430, set = 0, binding = 3) readonly buffer arrayMeta {
  float numbers[];
} meta;

void main() {
  uint M = uint(meta.numbers[0]), N = uint(meta.numbers[1]), K = uint(meta.numbers[2]);
  uint x = uint(gl_GlobalInvocationID.x);
  uint y = uint(gl_GlobalInvocationID.y);
  if (x >= N || y >= M) {
    return;
  }
  float sum = 0.0;
  for(uint k=0;k<K;k++) {
    sum += array_a.numbers[y * K + k] * array_b.numbers[k * N + x];
  }
  float alpha = meta.numbers[3];
  array_c.numbers[x + y * N] = sum * alpha;
}
