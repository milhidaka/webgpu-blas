export const Shader = `
@group(0) @binding(0)
var<storage,read> array_a: array<f32>;

@group(0) @binding(1)
var<storage,read> array_b: array<f32>;

@group(0) @binding(2)
var<storage,read_write> array_c: array<f32>;

struct CMeta {
  M: f32,
  N: f32,
  K: f32,
  alpha: f32,
}

@group(0) @binding(3)
var<storage,read> cmeta: CMeta;

@compute @workgroup_size(8,8,1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>
) {
  var M: u32 = u32(cmeta.M);
  var N: u32 = u32(cmeta.N);
  var K: u32 = u32(cmeta.K);
  var x: u32 = global_id.x;
  var y: u32 = global_id.y;
  if (x >= N || y >= M) {
    return;
  }
  var sum: f32 = 0.0;
  for(var k: u32 = 0u; k < K; k = k + 1u) {
    sum = array_a[y * K + k] * array_b[k * N + x] + sum;
  }
  array_c[x + y * N] = sum * cmeta.alpha;
}
`;
