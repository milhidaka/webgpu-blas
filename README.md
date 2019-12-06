# webgpu-blas

Fast matrix-matrix multiplication on web browser using [WebGPU](https://gpuweb.github.io/gpuweb/), future web standard.

Currently supports Safari on macOS Catalina and iOS 13.
Chrome support is future work.

On iPhone11, matrix multiplication of two 1024x1024 matrices can be computed in about 60ms (35GFLOPS), including data transfer from / to CPU.

To run WebGPU, experimental feature WebGPU have to be enabled in Safari.

In macOS, menu bar -> Develop -> Experimental Features -> check "WebGPU"

In iOS 13, open Settings -> Safari -> Advanced -> Experimental Features -> Toggle "WebGPU"

<p float="left">
<img src="docs/images/ios-safari-webgpu-1.png" title="iOS13 enable WebGPU Step1" width="200px">
<img src="docs/images/ios-safari-webgpu-2.png" title="iOS13 enable WebGPU Step2" width="200px">
<img src="docs/images/ios-safari-webgpu-3.png" title="iOS13 enable WebGPU Step3" width="200px">
<img src="docs/images/ios-safari-webgpu-4.png" title="iOS13 enable WebGPU Step4" width="200px">
</p>

# Usage

Fetch `webgpublas.js` from [Releases](https://github.com/milhidaka/webgpu-blas/releases).

```javascript
// <script src="webgpublas.js"></script>
const [m, n, k] = [64, 64, 64];
const array_a = new Float32Array(m * k);//m*k row-major matrix
const array_b = new Float32Array(k * n);//k*n row-major matrix
// fill array_a, array_b
for (let i = 0; i < array_a.length; i++) {
  array_a[i] = Math.random();
}
for (let i = 0; i < array_b.length; i++) {
  array_b[i] = Math.random();
}
const alpha = 1.0;
const result = await webgpublas.sgemm(m, n, k, alpha, array_a, array_b);
console.log(result); // m*n row-major matrix (Float32Array)
```

# Limitation
## sgemm
- Input matrix "C" of ordinary blas is not yet supported.
- To use efficient implementation, the condition `m % 64 === 0 && n % 32 === 0 && k % 4 === 0 && alpha === 1.0` have to met.
- When the device / browser does not support WebGPU, fallback pure JavaScript implementation is used.

# Development

Test is not yet implemented.

## Setup
```
yarn
```

## Build

For npm package
```
yarn build
```

For webpack single js
```
yarn webpack
```
