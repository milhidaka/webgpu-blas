import { sgemm as sgemm_chrome } from "./sgemm_chrome";
import { sgemm as sgemm_fallback } from "./sgemm_fallback";

let useFallback = false;
let lastError: Error | null = null;
export async function sgemm(m: number, n: number, k: number, alpha: number, a: Float32Array, b: Float32Array, beta: number = 0.0, c?: Float32Array): Promise<Float32Array> {
  if (useFallback) {
    return sgemm_fallback(m, n, k, alpha, a, b, beta, c);
  }
  let result: Float32Array | null = null;
  try {
    result = await sgemm_chrome(m, n, k, alpha, a, b, beta, c);
  } catch (error) {
    lastError = error as Error;
    console.warn('Error using WebGPU; fallback to pure JavaScript', error);
  }
  if (result === null) {
    useFallback = true;
    return sgemm_fallback(m, n, k, alpha, a, b, beta, c);
  }
  return result;
}

export function getWebGPUError(): Error | null {
  return lastError;
}
