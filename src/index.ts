import { sgemm as sgemm_chrome } from "./sgemm_chrome";
import { sgemm as sgemm_wsl } from "./sgemm_wsl";
import {sgemm as sgemm_fallback} from "./sgemm_fallback";

const hasWebGPUSupport = !!navigator.gpu && !!(typeof GPUBufferUsage !== 'undefined');
const isSafari = navigator.userAgent.includes('Safari') && !navigator.userAgent.includes('Chrome');

// TODO: 初期化失敗時のfallback
export const sgemm = hasWebGPUSupport ? (isSafari ? sgemm_wsl : sgemm_chrome) : sgemm_fallback;
