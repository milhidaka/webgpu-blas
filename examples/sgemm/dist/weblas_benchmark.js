
function message(m, target) {
  document.getElementById(target).innerText += m + '\n';
}

function makeRandom(length) {
  const array = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    array[i] = Math.random();
  }

  return array;
}

function parseMNKTuples(s) {
  const shapes = [];//[[m,n,k]]
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
    const shapes = parseMNKTuples((document.getElementById('benchmark_shapes')).value)
    const alpha = 1.0;
    const runs = 10;
    for (const [m, n, k] of shapes) {
      const array_a = makeRandom(m * k);
      const array_b = makeRandom(k * n);
      // warmup
      // weblas does not need await
      weblas.sgemm(m, n, k, alpha, array_a, array_b);
      let timeSum = 0;
      let retSum = 0;
      for (let i = 0; i < runs; i++) {
        console.time('sgemm');
        const sgemmStartTime = performance.now();//[ms]
        const result = weblas.sgemm(m, n, k, alpha, array_a, array_b, 0.0);
        retSum += result[0];
        const sgemmEndTime = performance.now();
        console.timeEnd('sgemm');
        timeSum += sgemmEndTime - sgemmStartTime;
      }
      const avgTime = timeSum / runs;
      const flops = m * n * k * 2 * 1000 / avgTime / 1000000000;
      message(`Sgemm of (${m}x${k}),(${k}x${n}): average ${avgTime} ms (${runs} runs), ${flops.toFixed(2)} GFLOPS`, messageTarget);
      console.log('sum of result (to avoid optimization)', retSum);
    }
  } catch (ex) {
    alert(ex.message);
  }
}

window.addEventListener('load', () => {
  document.getElementById('run_benchmark').onclick = run_benchmark;
});
