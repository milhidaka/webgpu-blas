
export function sgemm(m: number, n: number, k: number, alpha: number, array_a: Float32Array, array_b: Float32Array): Float32Array {
  // To improve performance on WebGPU unsupported devices, use WebGL or WebAssembly
  const result = new Float32Array(m * n);
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let sum = 0.0;
      for (let j = 0; j < k; j++) {
        sum += array_a[row * k + j] * array_b[j * n + col];
      }
      result[row * n + col] = sum * alpha;
    }
  }
  return result;
}
