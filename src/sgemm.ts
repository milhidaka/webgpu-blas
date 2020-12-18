import { Shader as ShaderSgemmGeneric } from "./shader_sgemm_generic";

let polyfillgpu_called = false;
let isSafari = navigator.userAgent.includes('Safari') && !navigator.userAgent.includes('Chrome');

function polyfillgpu() {
  if (polyfillgpu_called) {
    return;
  }
  polyfillgpu_called = true;
  if (!navigator.gpu) {
    console.error('navigator.gpu not found');
    return;
  }

  if (!GPUBufferUsage) {
    console.error('GPUBufferUsage not found');
  }

  // Safari 13.0.3 -> Safari TP>92
  if (isSafari) {
    if (GPUBufferUsage.COPY_SRC === undefined) {
      GPUBufferUsage.COPY_DST = GPUBufferUsage.TRANSFER_DST;
      GPUBufferUsage.COPY_SRC = GPUBufferUsage.TRANSFER_SRC;
      window.GPUShaderStage = GPUShaderStageBit;
    }
  }

}

type ThreadGroupDim = 'x' | 'y' | 'z';

interface WebGPURunnerBufferInfo {
  index: number;
  name: string;
  length: number;
  input: boolean;
  output: boolean;
}

interface WebGPURunnerPipeline {
  bindGroupLayout: any;
  pipeline: any;
}

interface WebGPURunnerRequest {
  pipeline: WebGPURunnerPipeline;
  buffers: WebGPURunnerBufferInfo[];
  inputData: { [name: string]: Float32Array };
  threadGroups: { [key in ThreadGroupDim]: number };
}

interface WebGPURunnerResult {
  outputData: { [name: string]: Float32Array };
}

class WebGPURunner {
  private _initialized = false;
  private _device: any;
  private _glslang!: any;
  isSupportedDevice: boolean;
  pipelineCache: Map<string, WebGPURunnerPipeline>;
  constructor() {
    this.pipelineCache = new Map();
    this.isSupportedDevice = false;
  }

  async init() {
    if (this._initialized) {
      return;
    }
    polyfillgpu();
    try {
      const adapter = await navigator.gpu.requestAdapter();
      this._device = await adapter.requestDevice();
      this.isSupportedDevice = true;
    } catch (ex) {
      console.error('Unsupported device: ', ex.message);
    }
    this._initialized = true;
  }

  createPipeline(shader: string | Uint32Array, nBuffers: number): WebGPURunnerPipeline {
    const device = this._device;
    const bindings = [];
    for (let i = 0; i < nBuffers; i++) {
      bindings.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer"
      });
    }
    console.log('isSafari', isSafari);
    const bindGroupLayout = isSafari ? device.createBindGroupLayout({
      bindings
    }) : device.createBindGroupLayout({
      entries: bindings
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    const shaderModule = isSafari ? device.createShaderModule({ code: shader, isWHLSL: true }) :
      device.createShaderModule({ code: shader });
    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      computeStage: {
        module: shaderModule,
        entryPoint: "main"
      }
    });

    return { bindGroupLayout, pipeline };
  }

  async run(request: WebGPURunnerRequest): Promise<WebGPURunnerResult> {
    const device = this._device;
    let chromeOutputCopyInfo: { src: any, dst: any, size: number, name: string }[] = [];
    const buffers = request.buffers.map((bparam, i) => {
      if (i !== bparam.index) {
        throw new Error('request.buffers is not sorted in order of index');
      }
      let usage = GPUBufferUsage.STORAGE;
      if (isSafari && bparam.input) {
        usage |= GPUBufferUsage.MAP_WRITE;
      }
      if (isSafari && bparam.output) {
        usage |= GPUBufferUsage.MAP_READ;
      }
      if (!isSafari && bparam.output) {
        usage |= GPUBufferUsage.COPY_SRC;
      }

      const buf = device.createBuffer({
        mappedAtCreation: (!isSafari && bparam.input) ? true : false,
        size: bparam.length * Float32Array.BYTES_PER_ELEMENT,
        usage
      });
      if (!isSafari && bparam.output) {
        const dst = device.createBuffer({
          size: bparam.length * Float32Array.BYTES_PER_ELEMENT,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        chromeOutputCopyInfo.push({ src: buf, dst, size: bparam.length * Float32Array.BYTES_PER_ELEMENT, name: bparam.name });
      }
      return buf;
    });
    const bindGroup = isSafari ? device.createBindGroup({
      layout: request.pipeline.bindGroupLayout,
      bindings: request.buffers.map((bparam, i) => ({
        binding: i,
        resource: {
          buffer: buffers[i],
          size: bparam.length * Float32Array.BYTES_PER_ELEMENT
        }
      }))
    }): device.createBindGroup({
      layout: request.pipeline.bindGroupLayout,
      entries: request.buffers.map((bparam, i) => ({
        binding: i,
        resource: {
          buffer: buffers[i],
          size: bparam.length * Float32Array.BYTES_PER_ELEMENT
        }
      }))
    });

    for (let i = 0; i < request.buffers.length; i++) {
      const bparam = request.buffers[i];

      if (bparam.input) {
        const buffer = buffers[i];
        if (isSafari) {
          const buffer_ab = await buffer.mapWriteAsync();
          const buffer_mapped_array = new Float32Array(buffer_ab);
          const input_array = request.inputData[bparam.name];
          if (!input_array) {
            console.error(`input array '${bparam.name}' is not supplied.`);
            continue;
          }
          if (input_array.length !== buffer_mapped_array.length) {
            console.error(`length of input array '${bparam.name}' does not match GPU buffer (${input_array.length} !== ${buffer_mapped_array.length}).`);
            continue;
          }
          buffer_mapped_array.set(input_array);
          buffer.unmap();
        } else {
          const buffer_ab = buffer.getMappedRange();
          const buffer_mapped_array = new Float32Array(buffer_ab);
          const input_array = request.inputData[bparam.name];
          if (!input_array) {
            console.error(`input array '${bparam.name}' is not supplied.`);
            continue;
          }
          if (input_array.length !== buffer_mapped_array.length) {
            console.error(`length of input array '${bparam.name}' does not match GPU buffer (${input_array.length} !== ${buffer_mapped_array.length}).`);
            continue;
          }
          buffer_mapped_array.set(input_array);
          buffer.unmap();
        }
      }
    }

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setPipeline(request.pipeline.pipeline);
    passEncoder.dispatch(
      request.threadGroups.x,
      request.threadGroups.y,
      request.threadGroups.z
    );
    passEncoder.endPass();

    for (const chromeCopy of chromeOutputCopyInfo) {
      commandEncoder.copyBufferToBuffer(chromeCopy.src, 0, chromeCopy.dst, 0, chromeCopy.size);
    }

    if (isSafari) {
      device.getQueue().submit([commandEncoder.finish()]);
    } else {
      device.defaultQueue.submit([commandEncoder.finish()]);
    }

    const outputs: { [key: string]: Float32Array } = {};
    if (isSafari) {
      for (let i = 0; i < request.buffers.length; i++) {
        const bparam = request.buffers[i];
  
        if (bparam.output) {
          const buffer = buffers[i];
          const buffer_ab = await buffer.mapReadAsync();
          const buffer_mapped_array = new Float32Array(buffer_ab);
          const result_array = new Float32Array(buffer_mapped_array);
          buffer.unmap();
          outputs[bparam.name] = result_array;
        }
      }
  
      for (let i = 0; i < request.buffers.length; i++) {
        const buffer = buffers[i];
        buffer.destroy();
      }
    } else {
      for (const chromeCopy of chromeOutputCopyInfo) {
        await chromeCopy.dst.mapAsync(GPUMapMode.READ);
        const arrayBuffer = chromeCopy.dst.getMappedRange();
        const buffer_mapped_array = new Float32Array(arrayBuffer);
        const result_array = new Float32Array(buffer_mapped_array);
        chromeCopy.dst.unmap();
        chromeCopy.dst.destroy();
        outputs[chromeCopy.name] = result_array;
      }
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }

    const result: WebGPURunnerResult = {
      outputData: outputs,
    };

    return result;
  }
}

const runner = new WebGPURunner();

async function sgemm_block(m: number, n: number, k: number, alpha: number, a: Float32Array, b: Float32Array): Promise<Float32Array> {
  const shader = `
[numthreads(8, 8, 1)]
compute void main(constant float4[] array_a : register(u0),
                  constant float4[] array_b : register(u1),
                  device float4[] array_c : register(u2),
                  constant float[] meta : register(u3),
                  float3 dispatchThreadID : SV_DispatchThreadID)
{
  // threadgroups x: N / numthread.x / 8, y: M / numthread.y / 4
  uint M = uint(meta[0]), N = uint(meta[1]), K = uint(meta[2]);
  uint MD4 = uint(meta[3]), ND4 = uint(meta[4]), KD4 = uint(meta[5]);
  uint x = uint(dispatchThreadID.x);
  uint y = uint(dispatchThreadID.y);
  float4 sum0 = float4(0.0,0.0,0.0,0.0);
  float4 sum1 = float4(0.0,0.0,0.0,0.0);
  float4 sum2 = float4(0.0,0.0,0.0,0.0);
  float4 sum3 = float4(0.0,0.0,0.0,0.0);
  float4 sum10 = float4(0.0,0.0,0.0,0.0);
  float4 sum11 = float4(0.0,0.0,0.0,0.0);
  float4 sum12 = float4(0.0,0.0,0.0,0.0);
  float4 sum13 = float4(0.0,0.0,0.0,0.0);
  for(uint k=0;k<KD4;k++) {
    float4 arow0 = array_a[(y * 4 + 0) * KD4 + k];
    float4 arow1 = array_a[(y * 4 + 1) * KD4 + k];
    float4 arow2 = array_a[(y * 4 + 2) * KD4 + k];
    float4 arow3 = array_a[(y * 4 + 3) * KD4 + k];
    float4 brow;
    brow = array_b[(k * 4 + 0) * ND4 + x * 2 + 0];
    sum0 = mad(float4(arow0.x,arow0.x,arow0.x,arow0.x), brow, sum0);
    sum1 = mad(float4(arow1.x,arow1.x,arow1.x,arow1.x), brow, sum1);
    sum2 = mad(float4(arow2.x,arow2.x,arow2.x,arow2.x), brow, sum2);
    sum3 = mad(float4(arow3.x,arow3.x,arow3.x,arow3.x), brow, sum3);
    brow = array_b[(k * 4 + 0) * ND4 + x * 2 + 1];
    sum10 = mad(float4(arow0.x,arow0.x,arow0.x,arow0.x), brow, sum10);
    sum11 = mad(float4(arow1.x,arow1.x,arow1.x,arow1.x), brow, sum11);
    sum12 = mad(float4(arow2.x,arow2.x,arow2.x,arow2.x), brow, sum12);
    sum13 = mad(float4(arow3.x,arow3.x,arow3.x,arow3.x), brow, sum13);
    brow = array_b[(k * 4 + 1) * ND4 + x * 2 + 0];
    sum0 = mad(float4(arow0.y,arow0.y,arow0.y,arow0.y), brow, sum0);
    sum1 = mad(float4(arow1.y,arow1.y,arow1.y,arow1.y), brow, sum1);
    sum2 = mad(float4(arow2.y,arow2.y,arow2.y,arow2.y), brow, sum2);
    sum3 = mad(float4(arow3.y,arow3.y,arow3.y,arow3.y), brow, sum3);
    brow = array_b[(k * 4 + 1) * ND4 + x * 2 + 1];
    sum10 = mad(float4(arow0.y,arow0.y,arow0.y,arow0.y), brow, sum10);
    sum11 = mad(float4(arow1.y,arow1.y,arow1.y,arow1.y), brow, sum11);
    sum12 = mad(float4(arow2.y,arow2.y,arow2.y,arow2.y), brow, sum12);
    sum13 = mad(float4(arow3.y,arow3.y,arow3.y,arow3.y), brow, sum13);
    brow = array_b[(k * 4 + 2) * ND4 + x * 2 + 0];
    sum0 = mad(float4(arow0.z,arow0.z,arow0.z,arow0.z), brow, sum0);
    sum1 = mad(float4(arow1.z,arow1.z,arow1.z,arow1.z), brow, sum1);
    sum2 = mad(float4(arow2.z,arow2.z,arow2.z,arow2.z), brow, sum2);
    sum3 = mad(float4(arow3.z,arow3.z,arow3.z,arow3.z), brow, sum3);
    brow = array_b[(k * 4 + 2) * ND4 + x * 2 + 1];
    sum10 = mad(float4(arow0.z,arow0.z,arow0.z,arow0.z), brow, sum10);
    sum11 = mad(float4(arow1.z,arow1.z,arow1.z,arow1.z), brow, sum11);
    sum12 = mad(float4(arow2.z,arow2.z,arow2.z,arow2.z), brow, sum12);
    sum13 = mad(float4(arow3.z,arow3.z,arow3.z,arow3.z), brow, sum13);
    brow = array_b[(k * 4 + 3) * ND4 + x * 2 + 0];
    sum0 = mad(float4(arow0.w,arow0.w,arow0.w,arow0.w), brow, sum0);
    sum1 = mad(float4(arow1.w,arow1.w,arow1.w,arow1.w), brow, sum1);
    sum2 = mad(float4(arow2.w,arow2.w,arow2.w,arow2.w), brow, sum2);
    sum3 = mad(float4(arow3.w,arow3.w,arow3.w,arow3.w), brow, sum3);
    brow = array_b[(k * 4 + 3) * ND4 + x * 2 + 1];
    sum10 = mad(float4(arow0.w,arow0.w,arow0.w,arow0.w), brow, sum10);
    sum11 = mad(float4(arow1.w,arow1.w,arow1.w,arow1.w), brow, sum11);
    sum12 = mad(float4(arow2.w,arow2.w,arow2.w,arow2.w), brow, sum12);
    sum13 = mad(float4(arow3.w,arow3.w,arow3.w,arow3.w), brow, sum13);
  }
    array_c[x * 2 + 0 + (y * 4 + 0) * ND4] = sum0;
    array_c[x * 2 + 0 + (y * 4 + 1) * ND4] = sum1;
    array_c[x * 2 + 0 + (y * 4 + 2) * ND4] = sum2;
    array_c[x * 2 + 0 + (y * 4 + 3) * ND4] = sum3;
    array_c[x * 2 + 1 + (y * 4 + 0) * ND4] = sum10;
    array_c[x * 2 + 1 + (y * 4 + 1) * ND4] = sum11;
    array_c[x * 2 + 1 + (y * 4 + 2) * ND4] = sum12;
    array_c[x * 2 + 1 + (y * 4 + 3) * ND4] = sum13;
}
`;

  const cache_key = 'sgemm_block';
  let pipeline = runner.pipelineCache.get(cache_key);
  if (!pipeline) {
    pipeline = runner.createPipeline(shader, 4);
    runner.pipelineCache.set(cache_key, pipeline);
  }

  const request: WebGPURunnerRequest = {
    pipeline,
    buffers: [
      { index: 0, name: 'array_a', length: m * k, input: true, output: false },
      { index: 1, name: 'array_b', length: k * n, input: true, output: false },
      { index: 2, name: 'array_c', length: m * n, input: false, output: true },
      { index: 3, name: 'meta', length: 6, input: true, output: false },
    ],
    inputData: { array_a: a, array_b: b, meta: new Float32Array([m, n, k, m / 4, n / 4, k / 4]) },
    threadGroups: { x: n / 64, y: m / 32, z: 1 }
  };

  const result = await runner.run(request);
  return result.outputData.array_c;
}

async function sgemm_generic(m: number, n: number, k: number, alpha: number, a: Float32Array, b: Float32Array): Promise<Float32Array> {
  const shader = isSafari ? `
[numthreads(8, 8, 1)]
compute void main(constant float[] array_a : register(u0),
                  constant float[] array_b : register(u1),
                  device float[] array_c : register(u2),
                  constant float[] meta : register(u3),
                  float3 dispatchThreadID : SV_DispatchThreadID)
{
  // threadgroups x: M / numthread.x, y: N / numthread.y
  uint M = uint(meta[0]), N = uint(meta[1]), K = uint(meta[2]);
  uint x = uint(dispatchThreadID.x);
  uint y = uint(dispatchThreadID.y);
  if (x >= N || y >= M) {
    return;
  }
  float sum = 0.0;
  for(uint k=0;k<K;k++) {
    sum += array_a[y * K + k] * array_b[k * N + x];
  }
  float alpha = meta[3];
    array_c[x + y * N] = sum * alpha;
}
` : ShaderSgemmGeneric;

  const cache_key = 'sgemm_generic';
  let pipeline = runner.pipelineCache.get(cache_key);
  if (!pipeline) {
    pipeline = runner.createPipeline(shader, 4);
    runner.pipelineCache.set(cache_key, pipeline);
  }

  const request: WebGPURunnerRequest = {
    pipeline,
    buffers: [
      { index: 0, name: 'array_a', length: m * k, input: true, output: false },
      { index: 1, name: 'array_b', length: k * n, input: true, output: false },
      { index: 2, name: 'array_c', length: m * n, input: false, output: true },
      { index: 3, name: 'meta', length: 4, input: true, output: false },
    ],
    inputData: { array_a: a, array_b: b, meta: new Float32Array([m, n, k, alpha]) },
    //threadGroups: isSafari ? { x: Math.ceil(n / 8), y: Math.ceil(m / 8), z: 1 } : { x: n, y: m, z: 1 }
    threadGroups: { x: n, y: m, z: 1 }
  };

  const result = await runner.run(request);
  return result.outputData.array_c;
}


function sgemm_fallback(m: number, n: number, k: number, alpha: number, array_a: Float32Array, array_b: Float32Array): Float32Array {
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

export async function sgemm(m: number, n: number, k: number, alpha: number, a: Float32Array, b: Float32Array, beta: number = 0.0, c?: Float32Array): Promise<Float32Array> {
  if (beta !== 0.0) {
    throw new Error('beta !== 0.0 is not yet supported');
  }

  await runner.init();
  if (!runner.isSupportedDevice) {
    // do fallback
    return sgemm_fallback(n, n, k, alpha, a, b);
  }

  if (m % 64 === 0 && n % 32 === 0 && k % 4 === 0 && alpha === 1.0) {
    return sgemm_block(m, n, k, alpha, a, b);
  } else {
    return sgemm_generic(m, n, k, alpha, a, b);
  }
}
