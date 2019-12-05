
function polyfillgpu() {
  if (!navigator.gpu) {
    alert('navigator.gpu not found');
    throw new Error('navigator.gpu not found');
  }

  if (!GPUBufferUsage) {
    alert('GPUBufferUsage not found');
    throw new Error('GPUBufferUsage not found');
  }

  // Safari 13.0.3 -> Safari TP>92
  if (GPUBufferUsage.COPY_SRC === undefined) {
    GPUBufferUsage.COPY_DST = GPUBufferUsage.TRANSFER_DST;
    GPUBufferUsage.COPY_SRC = GPUBufferUsage.TRANSFER_SRC;
    window.GPUShaderStage = GPUShaderStageBit;
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

interface WebGPURunnerRequest {
  shader: string;
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
  constructor() {
  }

  async init() {
    if (this._initialized) {
      return;
    }
    polyfillgpu();
    const adapter = await navigator.gpu.requestAdapter();
    this._device = await adapter.requestDevice();
    this._initialized = true;
  }

  async run(request: WebGPURunnerRequest): Promise<WebGPURunnerResult> {

    const device = this._device;
    const buffers = request.buffers.map((bparam, i) => {
      if (i !== bparam.index) {
        throw new Error('request.buffers is not sorted in order of index');
      }
      let usage = GPUBufferUsage.STORAGE;
      if (bparam.input) {
        usage |= GPUBufferUsage.MAP_WRITE;
      }
      if (bparam.output) {
        usage |= GPUBufferUsage.MAP_READ;
      }

      return device.createBuffer({
        size: bparam.length * Float32Array.BYTES_PER_ELEMENT,
        usage
      });
    });

    const bindGroupLayout = device.createBindGroupLayout({
      bindings: request.buffers.map((bparam, i) => ({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        type: "storage-buffer"
      }))
    });


    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      bindings: request.buffers.map((bparam, i) => ({
        binding: i,
        resource: {
          buffer: buffers[i],
          size: bparam.length * Float32Array.BYTES_PER_ELEMENT
        }
      }))
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    const shaderModule = device.createShaderModule({ code: request.shader, isWHLSL: true });

    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      computeStage: {
        module: shaderModule,
        entryPoint: "main"
      }
    });
    for (let i = 0; i < request.buffers.length; i++) {
      const bparam = request.buffers[i];

      if (bparam.input) {
        const buffer = buffers[i];
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
      }
    }

    const timeReadStart = Date.now();
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setPipeline(pipeline);
    passEncoder.dispatch(
      request.threadGroups.x,
      request.threadGroups.y,
      request.threadGroups.z
    );
    passEncoder.endPass();

    device.getQueue().submit([commandEncoder.finish()]);

    const outputs: { [key: string]: Float32Array } = {};
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

    const result: WebGPURunnerResult = {
      outputData: outputs,
    };

    return result;
  }
}

const runner = new WebGPURunner();

export async function sgemm(m: number, n: number, k: number, alpha: number, a: Float32Array, b: Float32Array, beta: number = 0.0, c?: Float32Array): Promise<Float32Array> {
  if (alpha !== 1.0) {
    throw new Error('alpha !== 1.0 is not yet supported');
  }
  if (beta !== 0.0) {
    throw new Error('beta !== 0.0 is not yet supported');
  }
  if (m % 64 !== 0) {
    throw new Error('m must be multiple of 64.');
  }
  if (n % 32 !== 0) {
    throw new Error('n must be multiple of 32.');
  }
  if (k % 4 !== 0) {
    throw new Error('n must be multiple of 4.');
  }

  await runner.init();

  const shader = `
[numthreads(8, 8, 1)]
compute void main(constant float4[] array_a : register(u0),
                  constant float4[] array_b : register(u1),
                  device float4[] array_c : register(u2),
                  constant float[] meta : register(u3),
                  float3 dispatchThreadID : SV_DispatchThreadID)
{
  // threadgroups x: M / numthread.x / 8, y: N / numthread.y / 4
  uint M = uint(meta[0]), N = uint(meta[1]), K = uint(meta[2]);
  uint MD4 = uint(meta[3]), ND4 = uint(meta[4]), KD4 = uint(meta[5]);
  //uint M = 64, N = 64, K = 64, MD4 = 16, ND4 = 16, KD4 = 16;
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

  const request: WebGPURunnerRequest = {
    shader,
    buffers: [
      { index: 0, name: 'array_a', length: m * k, input: true, output: false },
      { index: 1, name: 'array_b', length: k * n, input: true, output: false },
      { index: 2, name: 'array_c', length: m * n, input: false, output: true },
      { index: 3, name: 'meta', length: 6, input: true, output: false },
    ],
    inputData: { array_a: a, array_b: b, meta: new Float32Array([m, n, k, m / 4, n / 4, k / 4]) },
    threadGroups: { x: m / 64, y: n / 32, z: 1 }
  };

  const result = await runner.run(request);
  return result.outputData.array_c;
}
