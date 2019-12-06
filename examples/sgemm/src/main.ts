import { sgemm } from "webgpu-blas";

function message(m: string): void {
  document.getElementById('message').innerText += m + '\n';
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

async function compute() {
  try {
    const [m, n, k, alpha] = [
      Number((document.getElementById('size_m') as HTMLInputElement).value),
      Number((document.getElementById('size_n') as HTMLInputElement).value),
      Number((document.getElementById('size_k') as HTMLInputElement).value),
      Number((document.getElementById('arg_alpha') as HTMLInputElement).value)
    ];
    const array_a = makeRandom(m * k);
    const array_b = makeRandom(k * n);
    console.time('sgemm');
    const sgemmStartTime = performance.now();//[ms]
    const result = await sgemm(m, n, k, alpha, array_a, array_b);
    const sgemmEndTime = performance.now();
    console.timeEnd('sgemm');
    const flops = m * n * k * 2 * 1000 / (sgemmEndTime - sgemmStartTime) / 1000000000;
    message(`Sgemm of (${m}x${k}),(${k}x${n}): ${sgemmEndTime - sgemmStartTime} ms, ${flops.toFixed(2)} GFLOPS`);
    console.log('result', result);
    if ((document.getElementById('enable_validate') as HTMLInputElement).checked) {
      const validation_result = checkResult(m, n, k, alpha, array_a, array_b, result);
      console.log('validation result', validation_result);
      message(`Validation ${validation_result}`);
    }
  } catch (ex) {
    alert(ex.message);
  }
}

window.addEventListener('load', () => {
  document.getElementById('compute').onclick = compute;
});
