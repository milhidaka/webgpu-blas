#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) readonly buffer arrayA {
    vec4 numbers[];
} array_a;

layout(std430, set = 0, binding = 1) readonly buffer arrayB {
    vec4 numbers[];
} array_b;

layout(std430, set = 0, binding = 2) buffer arrayC {
    vec4 numbers[];
} array_c;

layout(std430, set = 0, binding = 3) readonly buffer arrayMeta {
  float numbers[];
} meta;

void main() {
  uint M = uint(meta.numbers[0]), N = uint(meta.numbers[1]), K = uint(meta.numbers[2]);
  uint MD4 = uint(meta.numbers[3]), ND4 = uint(meta.numbers[4]), KD4 = uint(meta.numbers[5]);
  uint x = uint(gl_GlobalInvocationID.x);
  uint y = uint(gl_GlobalInvocationID.y);

  vec4 sum0 = vec4(0.0,0.0,0.0,0.0);
  vec4 sum1 = vec4(0.0,0.0,0.0,0.0);
  vec4 sum2 = vec4(0.0,0.0,0.0,0.0);
  vec4 sum3 = vec4(0.0,0.0,0.0,0.0);
  vec4 sum10 = vec4(0.0,0.0,0.0,0.0);
  vec4 sum11 = vec4(0.0,0.0,0.0,0.0);
  vec4 sum12 = vec4(0.0,0.0,0.0,0.0);
  vec4 sum13 = vec4(0.0,0.0,0.0,0.0);
  for(uint k=0;k<KD4;k++) {
    vec4 arow0 = array_a.numbers[(y * 4 + 0) * KD4 + k];
    vec4 arow1 = array_a.numbers[(y * 4 + 1) * KD4 + k];
    vec4 arow2 = array_a.numbers[(y * 4 + 2) * KD4 + k];
    vec4 arow3 = array_a.numbers[(y * 4 + 3) * KD4 + k];
    vec4 brow;
    brow = array_b.numbers[(k * 4 + 0) * ND4 + x * 2 + 0];
    sum0 = (vec4(arow0.x,arow0.x,arow0.x,arow0.x) * brow + sum0);
    sum1 = (vec4(arow1.x,arow1.x,arow1.x,arow1.x) * brow + sum1);
    sum2 = (vec4(arow2.x,arow2.x,arow2.x,arow2.x) * brow + sum2);
    sum3 = (vec4(arow3.x,arow3.x,arow3.x,arow3.x) * brow + sum3);
    brow = array_b.numbers[(k * 4 + 0) * ND4 + x * 2 + 1];
    sum10 = (vec4(arow0.x,arow0.x,arow0.x,arow0.x) * brow + sum10);
    sum11 = (vec4(arow1.x,arow1.x,arow1.x,arow1.x) * brow + sum11);
    sum12 = (vec4(arow2.x,arow2.x,arow2.x,arow2.x) * brow + sum12);
    sum13 = (vec4(arow3.x,arow3.x,arow3.x,arow3.x) * brow + sum13);
    brow = array_b.numbers[(k * 4 + 1) * ND4 + x * 2 + 0];
    sum0 = (vec4(arow0.y,arow0.y,arow0.y,arow0.y) * brow + sum0);
    sum1 = (vec4(arow1.y,arow1.y,arow1.y,arow1.y) * brow + sum1);
    sum2 = (vec4(arow2.y,arow2.y,arow2.y,arow2.y) * brow + sum2);
    sum3 = (vec4(arow3.y,arow3.y,arow3.y,arow3.y) * brow + sum3);
    brow = array_b.numbers[(k * 4 + 1) * ND4 + x * 2 + 1];
    sum10 = (vec4(arow0.y,arow0.y,arow0.y,arow0.y) * brow + sum10);
    sum11 = (vec4(arow1.y,arow1.y,arow1.y,arow1.y) * brow + sum11);
    sum12 = (vec4(arow2.y,arow2.y,arow2.y,arow2.y) * brow + sum12);
    sum13 = (vec4(arow3.y,arow3.y,arow3.y,arow3.y) * brow + sum13);
    brow = array_b.numbers[(k * 4 + 2) * ND4 + x * 2 + 0];
    sum0 = (vec4(arow0.z,arow0.z,arow0.z,arow0.z) * brow + sum0);
    sum1 = (vec4(arow1.z,arow1.z,arow1.z,arow1.z) * brow + sum1);
    sum2 = (vec4(arow2.z,arow2.z,arow2.z,arow2.z) * brow + sum2);
    sum3 = (vec4(arow3.z,arow3.z,arow3.z,arow3.z) * brow + sum3);
    brow = array_b.numbers[(k * 4 + 2) * ND4 + x * 2 + 1];
    sum10 = (vec4(arow0.z,arow0.z,arow0.z,arow0.z) * brow + sum10);
    sum11 = (vec4(arow1.z,arow1.z,arow1.z,arow1.z) * brow + sum11);
    sum12 = (vec4(arow2.z,arow2.z,arow2.z,arow2.z) * brow + sum12);
    sum13 = (vec4(arow3.z,arow3.z,arow3.z,arow3.z) * brow + sum13);
    brow = array_b.numbers[(k * 4 + 3) * ND4 + x * 2 + 0];
    sum0 = (vec4(arow0.w,arow0.w,arow0.w,arow0.w) * brow + sum0);
    sum1 = (vec4(arow1.w,arow1.w,arow1.w,arow1.w) * brow + sum1);
    sum2 = (vec4(arow2.w,arow2.w,arow2.w,arow2.w) * brow + sum2);
    sum3 = (vec4(arow3.w,arow3.w,arow3.w,arow3.w) * brow + sum3);
    brow = array_b.numbers[(k * 4 + 3) * ND4 + x * 2 + 1];
    sum10 = (vec4(arow0.w,arow0.w,arow0.w,arow0.w) * brow + sum10);
    sum11 = (vec4(arow1.w,arow1.w,arow1.w,arow1.w) * brow + sum11);
    sum12 = (vec4(arow2.w,arow2.w,arow2.w,arow2.w) * brow + sum12);
    sum13 = (vec4(arow3.w,arow3.w,arow3.w,arow3.w) * brow + sum13);
  }
    array_c.numbers[x * 2 + 0 + (y * 4 + 0) * ND4] = sum0;
    array_c.numbers[x * 2 + 0 + (y * 4 + 1) * ND4] = sum1;
    array_c.numbers[x * 2 + 0 + (y * 4 + 2) * ND4] = sum2;
    array_c.numbers[x * 2 + 0 + (y * 4 + 3) * ND4] = sum3;
    array_c.numbers[x * 2 + 1 + (y * 4 + 0) * ND4] = sum10;
    array_c.numbers[x * 2 + 1 + (y * 4 + 1) * ND4] = sum11;
    array_c.numbers[x * 2 + 1 + (y * 4 + 2) * ND4] = sum12;
    array_c.numbers[x * 2 + 1 + (y * 4 + 3) * ND4] = sum13;
}
