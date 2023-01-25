export const Shader = `
@group(0) @binding(0)
var<storage,read> array_a: array<vec4<f32>>;

@group(0) @binding(1)
var<storage,read> array_b: array<vec4<f32>>;

@group(0) @binding(2)
var<storage,read_write> array_c: array<vec4<f32>>;

struct CMeta {
  M: f32,
  N: f32,
  K: f32,
  MD4: f32,
  ND4: f32,
  KD4: f32,
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
  var MD4: u32 = u32(cmeta.KD4);
  var ND4: u32 = u32(cmeta.ND4);
  var KD4: u32 = u32(cmeta.KD4);
  var x: u32 = global_id.x;
  var y: u32 = global_id.y;
  if (x >= N || y >= M) {
    return;
  }
  var alpha: f32 = cmeta.alpha;
  var sum00: vec4<f32> = vec4<f32>();
  var sum01: vec4<f32> = vec4<f32>();
  var sum02: vec4<f32> = vec4<f32>();
  var sum03: vec4<f32> = vec4<f32>();
  var sum10: vec4<f32> = vec4<f32>();
  var sum11: vec4<f32> = vec4<f32>();
  var sum12: vec4<f32> = vec4<f32>();
  var sum13: vec4<f32> = vec4<f32>();
  for(var k: u32 = 0u; k < KD4; k = k + 1u) {
    var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];
    var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];
    var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];
    var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];
    var brow: vec4<f32>;
    brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];
    sum00 = vec4<f32>(arow0.x) * brow + sum00;
    sum01 = vec4<f32>(arow1.x) * brow + sum01;
    sum02 = vec4<f32>(arow2.x) * brow + sum02;
    sum03 = vec4<f32>(arow3.x) * brow + sum03;
    brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];
    sum10 = vec4<f32>(arow0.x) * brow + sum10;
    sum11 = vec4<f32>(arow1.x) * brow + sum11;
    sum12 = vec4<f32>(arow2.x) * brow + sum12;
    sum13 = vec4<f32>(arow3.x) * brow + sum13;
    
    brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];
    sum00 = vec4<f32>(arow0.y) * brow + sum00;
    sum01 = vec4<f32>(arow1.y) * brow + sum01;
    sum02 = vec4<f32>(arow2.y) * brow + sum02;
    sum03 = vec4<f32>(arow3.y) * brow + sum03;
    brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];
    sum10 = vec4<f32>(arow0.y) * brow + sum10;
    sum11 = vec4<f32>(arow1.y) * brow + sum11;
    sum12 = vec4<f32>(arow2.y) * brow + sum12;
    sum13 = vec4<f32>(arow3.y) * brow + sum13;
    
    brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];
    sum00 = vec4<f32>(arow0.z) * brow + sum00;
    sum01 = vec4<f32>(arow1.z) * brow + sum01;
    sum02 = vec4<f32>(arow2.z) * brow + sum02;
    sum03 = vec4<f32>(arow3.z) * brow + sum03;
    brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];
    sum10 = vec4<f32>(arow0.z) * brow + sum10;
    sum11 = vec4<f32>(arow1.z) * brow + sum11;
    sum12 = vec4<f32>(arow2.z) * brow + sum12;
    sum13 = vec4<f32>(arow3.z) * brow + sum13;
    
    brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];
    sum00 = vec4<f32>(arow0.w) * brow + sum00;
    sum01 = vec4<f32>(arow1.w) * brow + sum01;
    sum02 = vec4<f32>(arow2.w) * brow + sum02;
    sum03 = vec4<f32>(arow3.w) * brow + sum03;
    brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];
    sum10 = vec4<f32>(arow0.w) * brow + sum10;
    sum11 = vec4<f32>(arow1.w) * brow + sum11;
    sum12 = vec4<f32>(arow2.w) * brow + sum12;
    sum13 = vec4<f32>(arow3.w) * brow + sum13;
  }
  array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00 * alpha;
  array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01 * alpha;
  array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02 * alpha;
  array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03 * alpha;
  array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10 * alpha;
  array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11 * alpha;
  array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12 * alpha;
  array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13 * alpha;
}
`;
