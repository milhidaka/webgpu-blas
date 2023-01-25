import { Shader as ShaderSgemmBlock } from "./shader_sgemm_block";
import { Shader as ShaderSgemmGeneric } from "./shader_sgemm_generic";

let polyfillgpu_called = false;

function polyfillgpu() {
  if (polyfillgpu_called) {
    return;
  }
  polyfillgpu_called = true;
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
  private _device!: GPUDevice;
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
      if (!adapter) {
        throw new Error("requestAdapter failed");
      }
      this._device = await adapter.requestDevice();
      this.isSupportedDevice = true;
    } catch (ex) {
      console.error('Unsupported device: ', (ex as Error).message);
    }
    this._initialized = true;
  }

  createPipeline(shader: string, bindingTypes: GPUBufferBindingType[]): WebGPURunnerPipeline {
    const device = this._device;
    const bindings: GPUBindGroupLayoutEntry[] = [];
    for (let i = 0; i < bindingTypes.length; i++) {
      bindings.push({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: bindingTypes[i] },
      });
    }
    const bindGroupLayout = device.createBindGroupLayout({
      entries: bindings
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

    const shaderModule = device.createShaderModule({ code: shader });
    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
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
      if (bparam.output) {
        usage |= GPUBufferUsage.COPY_SRC;
      }

      const buf = device.createBuffer({
        mappedAtCreation: bparam.input ? true : false,
        size: bparam.length * Float32Array.BYTES_PER_ELEMENT,
        usage
      });
      if (bparam.output) {
        const dst = device.createBuffer({
          size: bparam.length * Float32Array.BYTES_PER_ELEMENT,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        chromeOutputCopyInfo.push({ src: buf, dst, size: bparam.length * Float32Array.BYTES_PER_ELEMENT, name: bparam.name });
      }
      return buf;
    });
    const bindGroup = device.createBindGroup({
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

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setPipeline(request.pipeline.pipeline);
    passEncoder.dispatchWorkgroups(
      request.threadGroups.x,
      request.threadGroups.y,
      request.threadGroups.z
    );
    if (passEncoder.end) {
      passEncoder.end();
    } else {
      // deprecated
      // Firefox Nightly 111 has this
      (passEncoder as any).endPass();
    }

    for (const chromeCopy of chromeOutputCopyInfo) {
      commandEncoder.copyBufferToBuffer(chromeCopy.src, 0, chromeCopy.dst, 0, chromeCopy.size);
    }

    device.queue.submit([commandEncoder.finish()]);

    const outputs: { [key: string]: Float32Array } = {};
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


    const result: WebGPURunnerResult = {
      outputData: outputs,
    };

    return result;
  }
}

const runner = new WebGPURunner();

async function sgemm_block(m: number, n: number, k: number, alpha: number, a: Float32Array, b: Float32Array): Promise<Float32Array> {
  const shader = ShaderSgemmBlock;

  const cache_key = 'sgemm_block';
  let pipeline = runner.pipelineCache.get(cache_key);
  if (!pipeline) {
    pipeline = runner.createPipeline(shader, ['read-only-storage', 'read-only-storage', 'storage', 'read-only-storage']);
    runner.pipelineCache.set(cache_key, pipeline);
  }

  const request: WebGPURunnerRequest = {
    pipeline,
    buffers: [
      { index: 0, name: 'array_a', length: m * k, input: true, output: false },
      { index: 1, name: 'array_b', length: k * n, input: true, output: false },
      { index: 2, name: 'array_c', length: m * n, input: false, output: true },
      { index: 3, name: 'meta', length: 7, input: true, output: false },
    ],
    inputData: { array_a: a, array_b: b, meta: new Float32Array([m, n, k, m / 4, n / 4, k / 4, alpha]) },
    threadGroups: { x: n / 64, y: m / 32, z: 1 }
  };

  const result = await runner.run(request);
  return result.outputData.array_c;
}

async function sgemm_generic(m: number, n: number, k: number, alpha: number, a: Float32Array, b: Float32Array): Promise<Float32Array> {
  const shader = ShaderSgemmGeneric;

  const cache_key = 'sgemm_generic';
  let pipeline = runner.pipelineCache.get(cache_key);
  if (!pipeline) {
    pipeline = runner.createPipeline(shader, ['read-only-storage', 'read-only-storage', 'storage', 'read-only-storage']);
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
    threadGroups: { x: Math.ceil(n / 8), y: Math.ceil(m / 8), z: 1 }
  };

  const result = await runner.run(request);
  return result.outputData.array_c;
}



export async function sgemm(m: number, n: number, k: number, alpha: number, a: Float32Array, b: Float32Array, beta: number = 0.0, c?: Float32Array): Promise<Float32Array> {
  if (beta !== 0.0) {
    throw new Error('beta !== 0.0 is not yet supported');
  }

  await runner.init();
  if (!runner.isSupportedDevice) {
    // do fallback
    throw new Error('unsupported device');
  }

  if (m % 32 === 0 && n % 64 === 0 && k % 4 === 0 && alpha === 1.0) {
    return sgemm_block(m, n, k, alpha, a, b);
  } else {
    return sgemm_generic(m, n, k, alpha, a, b);
  }
}
