import { sgemm } from "webgpu-blas";

function makeRandom(length: number): Float32Array {
  const array = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    array[i] = Math.random();
  }

  return array;
}

async function compute() {
  const [m, n, k] = [64, 64, 64];
  const array_a = makeRandom(m * k);
  const array_b = makeRandom(k * n);
  console.time('sgemm');
  const result = await sgemm(m, n, k, 1.0, array_a, array_b);
  console.timeEnd('sgemm');
  console.log('result', result[0]);
}

window.addEventListener('load', () => {
  document.getElementById('compute').onclick = compute;
});
