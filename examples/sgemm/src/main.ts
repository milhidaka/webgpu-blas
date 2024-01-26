import { sgemm, getWebGPUError } from "webgpu-blas";

let errorReported = false;

function alertIfError() {
  if (errorReported) {
    return;
  }

  let error = getWebGPUError();
  if (error) {
    alert(`WebGPU Error (fallback to pure JavaScript): ${error}`);
    errorReported = true;
  }
}

function message(m: string, target: string): void {
  document.getElementById(target).innerText += m + '\n';
}

function makeRandom(length: number): Float32Array {
  const array = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    array[i] = Math.random();
  }

  return array;
}

function checkResult(m: number, n: number, k: number, alpha: number, array_a: Float32Array, array_b: Float32Array, actual: Float32Array): boolean {
  const expected = new Float32Array(m * n);
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      let sum = 0.0;
      for (let j = 0; j < k; j++) {
        sum += array_a[row * k + j] * array_b[j * n + col];
      }
      expected[row * n + col] = sum * alpha;
    }
  }
  for (let row = 0; row < m; row++) {
    for (let col = 0; col < n; col++) {
      const idx = row * n + col;
      const expected_el = expected[idx];
      const actual_el = actual[idx];
      if (Math.abs(expected_el - actual_el) > (1e-5 + 1e-3 * Math.abs(expected_el))) {
        console.error(`[${row}, ${col}]: ${expected_el} !== ${actual_el}`);
        return false;
      }
    }
  }
  return true;
}

function parseMNKTuples(s: string): number[][] {
  const shapes: number[][] = [];//[[m,n,k]]
  for (const line of s.split('\n')) {
    const parts = line.split(',').map((t) => Number(t.trim()));
    if (parts.length === 3 && parts.every((v) => v > 0)) {
      shapes.push(parts);
    }
  }
  return shapes;
}

async function run_benchmark() {
  const messageTarget = 'bench_message';
  try {
    const shapes = parseMNKTuples((document.getElementById('benchmark_shapes') as HTMLTextAreaElement).value)
    const alpha = 1.0;
    const runs = 10;
    for (const [m, n, k] of shapes) {
      const array_a = makeRandom(m * k);
      const array_b = makeRandom(k * n);
      // warmup
      await sgemm(m, n, k, alpha, array_a, array_b);
      alertIfError();
      let timeSum = 0;
      let retSum = 0;
      for (let i = 0; i < runs; i++) {
        console.time('sgemm');
        const sgemmStartTime = performance.now();//[ms]
        const result = await sgemm(m, n, k, alpha, array_a, array_b);
        retSum += result[0];
        const sgemmEndTime = performance.now();
        console.timeEnd('sgemm');
        timeSum += sgemmEndTime - sgemmStartTime;
      }
      const avgTime = timeSum / runs;
      const flops = m * n * k * 2 * 1000 / avgTime / 1000000000;
      message(`Sgemm of (${m}x${k}),(${k}x${n}): average ${avgTime} ms (${runs} runs), ${flops.toFixed(2)} GFLOPS`, messageTarget);
      console.log('sum of result (to avoid optimization)', retSum);
      alertIfError();
    }
  } catch (ex) {
    alert(ex.message);
  }
}

async function small_example() {
  try {
    const array_a = new Float32Array([1, 2, 3, 4]);
    const array_b = new Float32Array([5, 6, 7, 8]);
    const result = await sgemm(2, 2, 2, 1, array_a, array_b);
    alertIfError();
    document.getElementById('small_example_result').innerText = `[${result[0]}, ${result[1]}\n ${result[2]}, ${result[3]}]`;
  } catch (ex) {
    alert(ex.message);
  }
}

async function run_test() {
  const shapes = parseMNKTuples((document.getElementById('test_shapes') as HTMLTextAreaElement).value)
  const alpha = 1.0;
  const messageTarget = 'test_message';
  for (const [m, n, k] of shapes) {
    const array_a = makeRandom(m * k);
    const array_b = makeRandom(k * n);
    const result = await sgemm(m, n, k, alpha, array_a, array_b);
    alertIfError();
    const validation_result = checkResult(m, n, k, alpha, array_a, array_b, result);
    message(`M=${m}, N=${n}, K=${k}: ${validation_result ? 'OK' : 'Error'}`, messageTarget);
  }
}

window.addEventListener('load', () => {
  document.getElementById('run_benchmark').onclick = run_benchmark;
  document.getElementById('small_example').onclick = small_example;
  document.getElementById('run_test').onclick = run_test;
  document.getElementById('is_webgpu_enabled').innerText = (navigator as any).gpu ? 'Enabled' : 'Disabled (fallback pure JavaScript implementation will be used)';
});
