import { sgemm as sgemm_chrome } from "./sgemm_chrome";
import { sgemm as sgemm_wsl } from "./sgemm_wsl";
import { sgemm as sgemm_fallback } from "./sgemm_fallback";

const isSafari = navigator.userAgent.includes('Safari') && !navigator.userAgent.includes('Chrome');

let useFallback = false;
export async function sgemm(m: number, n: number, k: number, alpha: number, a: Float32Array, b: Float32Array, beta: number = 0.0, c?: Float32Array): Promise<Float32Array> {
  if (useFallback) {
    return sgemm_fallback(m, n, k, alpha, a, b, beta, c);
  }
  let result: Float32Array | null = null;
  if (isSafari) {
    try {
      result = await sgemm_wsl(m, n, k, alpha, a, b, beta, c);
    } catch (error) {
      console.warn('Error using WebGPU; fallback to pure JavaScript', error);
    }
  } else {
    try {
      result = await sgemm_chrome(m, n, k, alpha, a, b, beta, c);
    } catch (error) {
      console.warn('Error using WebGPU; fallback to pure JavaScript', error);
    }
  }
  if (result === null) {
    useFallback = true;
    return sgemm_fallback(m, n, k, alpha, a, b, beta, c);
  }
  return result;
}
