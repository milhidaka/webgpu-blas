/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, { enumerable: true, get: getter });
/******/ 		}
/******/ 	};
/******/
/******/ 	// define __esModule on exports
/******/ 	__webpack_require__.r = function(exports) {
/******/ 		if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 			Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 		}
/******/ 		Object.defineProperty(exports, '__esModule', { value: true });
/******/ 	};
/******/
/******/ 	// create a fake namespace object
/******/ 	// mode & 1: value is a module id, require it
/******/ 	// mode & 2: merge all properties of value into the ns
/******/ 	// mode & 4: return value when already ns object
/******/ 	// mode & 8|1: behave like require
/******/ 	__webpack_require__.t = function(value, mode) {
/******/ 		if(mode & 1) value = __webpack_require__(value);
/******/ 		if(mode & 8) return value;
/******/ 		if((mode & 4) && typeof value === 'object' && value && value.__esModule) return value;
/******/ 		var ns = Object.create(null);
/******/ 		__webpack_require__.r(ns);
/******/ 		Object.defineProperty(ns, 'default', { enumerable: true, value: value });
/******/ 		if(mode & 2 && typeof value != 'string') for(var key in value) __webpack_require__.d(ns, key, function(key) { return value[key]; }.bind(null, key));
/******/ 		return ns;
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = "./src/main.ts");
/******/ })
/************************************************************************/
/******/ ({

/***/ "./node_modules/webgpu-blas/dist/index.js":
/*!************************************************!*\
  !*** ./node_modules/webgpu-blas/dist/index.js ***!
  \************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";
eval("\nvar __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {\n    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }\n    return new (P || (P = Promise))(function (resolve, reject) {\n        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }\n        function rejected(value) { try { step(generator[\"throw\"](value)); } catch (e) { reject(e); } }\n        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }\n        step((generator = generator.apply(thisArg, _arguments || [])).next());\n    });\n};\nObject.defineProperty(exports, \"__esModule\", { value: true });\nexports.sgemm = void 0;\nconst sgemm_chrome_1 = __webpack_require__(/*! ./sgemm_chrome */ \"./node_modules/webgpu-blas/dist/sgemm_chrome.js\");\nconst sgemm_fallback_1 = __webpack_require__(/*! ./sgemm_fallback */ \"./node_modules/webgpu-blas/dist/sgemm_fallback.js\");\nlet useFallback = false;\nfunction sgemm(m, n, k, alpha, a, b, beta = 0.0, c) {\n    return __awaiter(this, void 0, void 0, function* () {\n        if (useFallback) {\n            return sgemm_fallback_1.sgemm(m, n, k, alpha, a, b, beta, c);\n        }\n        let result = null;\n        try {\n            result = yield sgemm_chrome_1.sgemm(m, n, k, alpha, a, b, beta, c);\n        }\n        catch (error) {\n            console.warn('Error using WebGPU; fallback to pure JavaScript', error);\n        }\n        if (result === null) {\n            useFallback = true;\n            return sgemm_fallback_1.sgemm(m, n, k, alpha, a, b, beta, c);\n        }\n        return result;\n    });\n}\nexports.sgemm = sgemm;\n\n\n//# sourceURL=webpack:///./node_modules/webgpu-blas/dist/index.js?");

/***/ }),

/***/ "./node_modules/webgpu-blas/dist/sgemm_chrome.js":
/*!*******************************************************!*\
  !*** ./node_modules/webgpu-blas/dist/sgemm_chrome.js ***!
  \*******************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";
eval("\nvar __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {\n    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }\n    return new (P || (P = Promise))(function (resolve, reject) {\n        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }\n        function rejected(value) { try { step(generator[\"throw\"](value)); } catch (e) { reject(e); } }\n        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }\n        step((generator = generator.apply(thisArg, _arguments || [])).next());\n    });\n};\nObject.defineProperty(exports, \"__esModule\", { value: true });\nexports.sgemm = void 0;\nconst shader_sgemm_block_1 = __webpack_require__(/*! ./shader_sgemm_block */ \"./node_modules/webgpu-blas/dist/shader_sgemm_block.js\");\nconst shader_sgemm_generic_1 = __webpack_require__(/*! ./shader_sgemm_generic */ \"./node_modules/webgpu-blas/dist/shader_sgemm_generic.js\");\nlet polyfillgpu_called = false;\nfunction polyfillgpu() {\n    if (polyfillgpu_called) {\n        return;\n    }\n    polyfillgpu_called = true;\n}\nclass WebGPURunner {\n    constructor() {\n        this._initialized = false;\n        this.pipelineCache = new Map();\n        this.isSupportedDevice = false;\n    }\n    init() {\n        return __awaiter(this, void 0, void 0, function* () {\n            if (this._initialized) {\n                return;\n            }\n            polyfillgpu();\n            try {\n                const adapter = yield navigator.gpu.requestAdapter();\n                if (!adapter) {\n                    throw new Error(\"requestAdapter failed\");\n                }\n                this._device = yield adapter.requestDevice();\n                this.isSupportedDevice = true;\n            }\n            catch (ex) {\n                console.error('Unsupported device: ', ex.message);\n            }\n            this._initialized = true;\n        });\n    }\n    createPipeline(shader, nBuffers) {\n        const device = this._device;\n        const bindings = [];\n        for (let i = 0; i < nBuffers; i++) {\n            bindings.push({\n                binding: i,\n                visibility: GPUShaderStage.COMPUTE,\n                buffer: { type: \"storage\" },\n            });\n        }\n        const bindGroupLayout = device.createBindGroupLayout({\n            entries: bindings\n        });\n        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });\n        const shaderModule = device.createShaderModule({ code: shader });\n        const pipeline = device.createComputePipeline({\n            layout: pipelineLayout,\n            compute: {\n                module: shaderModule,\n                entryPoint: \"main\"\n            }\n        });\n        return { bindGroupLayout, pipeline };\n    }\n    run(request) {\n        return __awaiter(this, void 0, void 0, function* () {\n            const device = this._device;\n            let chromeOutputCopyInfo = [];\n            const buffers = request.buffers.map((bparam, i) => {\n                if (i !== bparam.index) {\n                    throw new Error('request.buffers is not sorted in order of index');\n                }\n                let usage = GPUBufferUsage.STORAGE;\n                if (bparam.output) {\n                    usage |= GPUBufferUsage.COPY_SRC;\n                }\n                const buf = device.createBuffer({\n                    mappedAtCreation: bparam.input ? true : false,\n                    size: bparam.length * Float32Array.BYTES_PER_ELEMENT,\n                    usage\n                });\n                if (bparam.output) {\n                    const dst = device.createBuffer({\n                        size: bparam.length * Float32Array.BYTES_PER_ELEMENT,\n                        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,\n                    });\n                    chromeOutputCopyInfo.push({ src: buf, dst, size: bparam.length * Float32Array.BYTES_PER_ELEMENT, name: bparam.name });\n                }\n                return buf;\n            });\n            const bindGroup = device.createBindGroup({\n                layout: request.pipeline.bindGroupLayout,\n                entries: request.buffers.map((bparam, i) => ({\n                    binding: i,\n                    resource: {\n                        buffer: buffers[i],\n                        size: bparam.length * Float32Array.BYTES_PER_ELEMENT\n                    }\n                }))\n            });\n            for (let i = 0; i < request.buffers.length; i++) {\n                const bparam = request.buffers[i];\n                if (bparam.input) {\n                    const buffer = buffers[i];\n                    const buffer_ab = buffer.getMappedRange();\n                    const buffer_mapped_array = new Float32Array(buffer_ab);\n                    const input_array = request.inputData[bparam.name];\n                    if (!input_array) {\n                        console.error(`input array '${bparam.name}' is not supplied.`);\n                        continue;\n                    }\n                    if (input_array.length !== buffer_mapped_array.length) {\n                        console.error(`length of input array '${bparam.name}' does not match GPU buffer (${input_array.length} !== ${buffer_mapped_array.length}).`);\n                        continue;\n                    }\n                    buffer_mapped_array.set(input_array);\n                    buffer.unmap();\n                }\n            }\n            const commandEncoder = device.createCommandEncoder();\n            const passEncoder = commandEncoder.beginComputePass();\n            passEncoder.setBindGroup(0, bindGroup);\n            passEncoder.setPipeline(request.pipeline.pipeline);\n            passEncoder.dispatch(request.threadGroups.x, request.threadGroups.y, request.threadGroups.z);\n            passEncoder.end();\n            for (const chromeCopy of chromeOutputCopyInfo) {\n                commandEncoder.copyBufferToBuffer(chromeCopy.src, 0, chromeCopy.dst, 0, chromeCopy.size);\n            }\n            device.queue.submit([commandEncoder.finish()]);\n            const outputs = {};\n            for (const chromeCopy of chromeOutputCopyInfo) {\n                yield chromeCopy.dst.mapAsync(GPUMapMode.READ);\n                const arrayBuffer = chromeCopy.dst.getMappedRange();\n                const buffer_mapped_array = new Float32Array(arrayBuffer);\n                const result_array = new Float32Array(buffer_mapped_array);\n                chromeCopy.dst.unmap();\n                chromeCopy.dst.destroy();\n                outputs[chromeCopy.name] = result_array;\n            }\n            for (const buffer of buffers) {\n                buffer.destroy();\n            }\n            const result = {\n                outputData: outputs,\n            };\n            return result;\n        });\n    }\n}\nconst runner = new WebGPURunner();\nfunction sgemm_block(m, n, k, alpha, a, b) {\n    return __awaiter(this, void 0, void 0, function* () {\n        const shader = shader_sgemm_block_1.Shader;\n        const cache_key = 'sgemm_block';\n        let pipeline = runner.pipelineCache.get(cache_key);\n        if (!pipeline) {\n            pipeline = runner.createPipeline(shader, 4);\n            runner.pipelineCache.set(cache_key, pipeline);\n        }\n        const request = {\n            pipeline,\n            buffers: [\n                { index: 0, name: 'array_a', length: m * k, input: true, output: false },\n                { index: 1, name: 'array_b', length: k * n, input: true, output: false },\n                { index: 2, name: 'array_c', length: m * n, input: false, output: true },\n                { index: 3, name: 'meta', length: 7, input: true, output: false },\n            ],\n            inputData: { array_a: a, array_b: b, meta: new Float32Array([m, n, k, m / 4, n / 4, k / 4, alpha]) },\n            threadGroups: { x: n / 64, y: m / 32, z: 1 }\n        };\n        const result = yield runner.run(request);\n        return result.outputData.array_c;\n    });\n}\nfunction sgemm_generic(m, n, k, alpha, a, b) {\n    return __awaiter(this, void 0, void 0, function* () {\n        const shader = shader_sgemm_generic_1.Shader;\n        const cache_key = 'sgemm_generic';\n        let pipeline = runner.pipelineCache.get(cache_key);\n        if (!pipeline) {\n            pipeline = runner.createPipeline(shader, 4);\n            runner.pipelineCache.set(cache_key, pipeline);\n        }\n        const request = {\n            pipeline,\n            buffers: [\n                { index: 0, name: 'array_a', length: m * k, input: true, output: false },\n                { index: 1, name: 'array_b', length: k * n, input: true, output: false },\n                { index: 2, name: 'array_c', length: m * n, input: false, output: true },\n                { index: 3, name: 'meta', length: 4, input: true, output: false },\n            ],\n            inputData: { array_a: a, array_b: b, meta: new Float32Array([m, n, k, alpha]) },\n            threadGroups: { x: Math.ceil(n / 8), y: Math.ceil(m / 8), z: 1 }\n        };\n        const result = yield runner.run(request);\n        return result.outputData.array_c;\n    });\n}\nfunction sgemm(m, n, k, alpha, a, b, beta = 0.0, c) {\n    return __awaiter(this, void 0, void 0, function* () {\n        if (beta !== 0.0) {\n            throw new Error('beta !== 0.0 is not yet supported');\n        }\n        yield runner.init();\n        if (!runner.isSupportedDevice) {\n            // do fallback\n            throw new Error('unsupported device');\n        }\n        if (m % 32 === 0 && n % 64 === 0 && k % 4 === 0 && alpha === 1.0) {\n            return sgemm_block(m, n, k, alpha, a, b);\n        }\n        else {\n            return sgemm_generic(m, n, k, alpha, a, b);\n        }\n    });\n}\nexports.sgemm = sgemm;\n\n\n//# sourceURL=webpack:///./node_modules/webgpu-blas/dist/sgemm_chrome.js?");

/***/ }),

/***/ "./node_modules/webgpu-blas/dist/sgemm_fallback.js":
/*!*********************************************************!*\
  !*** ./node_modules/webgpu-blas/dist/sgemm_fallback.js ***!
  \*********************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";
eval("\nObject.defineProperty(exports, \"__esModule\", { value: true });\nexports.sgemm = void 0;\nfunction sgemm(m, n, k, alpha, a, b, beta = 0.0, c) {\n    // To improve performance on WebGPU unsupported devices, use WebGL or WebAssembly\n    if (beta !== 0.0) {\n        throw new Error('beta !== 0.0 is not yet supported');\n    }\n    const result = new Float32Array(m * n);\n    for (let row = 0; row < m; row++) {\n        for (let col = 0; col < n; col++) {\n            let sum = 0.0;\n            for (let j = 0; j < k; j++) {\n                sum += a[row * k + j] * b[j * n + col];\n            }\n            result[row * n + col] = sum * alpha;\n        }\n    }\n    return result;\n}\nexports.sgemm = sgemm;\n\n\n//# sourceURL=webpack:///./node_modules/webgpu-blas/dist/sgemm_fallback.js?");

/***/ }),

/***/ "./node_modules/webgpu-blas/dist/shader_sgemm_block.js":
/*!*************************************************************!*\
  !*** ./node_modules/webgpu-blas/dist/shader_sgemm_block.js ***!
  \*************************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";
eval("\nObject.defineProperty(exports, \"__esModule\", { value: true });\nexports.Shader = void 0;\nexports.Shader = `\n@group(0) @binding(0)\nvar<storage,read> array_a: array<vec4<f32>>;\n\n@group(0) @binding(1)\nvar<storage,read> array_b: array<vec4<f32>>;\n\n@group(0) @binding(2)\nvar<storage,read_write> array_c: array<vec4<f32>>;\n\nstruct Meta {\n  M: f32;\n  N: f32;\n  K: f32;\n  MD4: f32;\n  ND4: f32;\n  KD4: f32;\n  alpha: f32;\n}\n\n@group(0) @binding(3)\nvar<storage,read> meta: Meta;\n\n@stage(compute) @workgroup_size(8,8,1)\nfn main(\n  @builtin(global_invocation_id) global_id: vec3<u32>\n) {\n  var M: u32 = u32(meta.M);\n  var N: u32 = u32(meta.N);\n  var K: u32 = u32(meta.K);\n  var MD4: u32 = u32(meta.KD4);\n  var ND4: u32 = u32(meta.ND4);\n  var KD4: u32 = u32(meta.KD4);\n  var x: u32 = global_id.x;\n  var y: u32 = global_id.y;\n  if (x >= N || y >= M) {\n    return;\n  }\n  var alpha: f32 = meta.alpha;\n  var sum00: vec4<f32> = vec4<f32>();\n  var sum01: vec4<f32> = vec4<f32>();\n  var sum02: vec4<f32> = vec4<f32>();\n  var sum03: vec4<f32> = vec4<f32>();\n  var sum10: vec4<f32> = vec4<f32>();\n  var sum11: vec4<f32> = vec4<f32>();\n  var sum12: vec4<f32> = vec4<f32>();\n  var sum13: vec4<f32> = vec4<f32>();\n  for(var k: u32 = 0u; k < KD4; k = k + 1u) {\n    var arow0: vec4<f32> = array_a[(y * 4u + 0u) * KD4 + k];\n    var arow1: vec4<f32> = array_a[(y * 4u + 1u) * KD4 + k];\n    var arow2: vec4<f32> = array_a[(y * 4u + 2u) * KD4 + k];\n    var arow3: vec4<f32> = array_a[(y * 4u + 3u) * KD4 + k];\n    var brow: vec4<f32>;\n    brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 0u];\n    sum00 = vec4<f32>(arow0.x) * brow + sum00;\n    sum01 = vec4<f32>(arow1.x) * brow + sum01;\n    sum02 = vec4<f32>(arow2.x) * brow + sum02;\n    sum03 = vec4<f32>(arow3.x) * brow + sum03;\n    brow = array_b[(k * 4u + 0u) * ND4 + x * 2u + 1u];\n    sum10 = vec4<f32>(arow0.x) * brow + sum10;\n    sum11 = vec4<f32>(arow1.x) * brow + sum11;\n    sum12 = vec4<f32>(arow2.x) * brow + sum12;\n    sum13 = vec4<f32>(arow3.x) * brow + sum13;\n    \n    brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 0u];\n    sum00 = vec4<f32>(arow0.y) * brow + sum00;\n    sum01 = vec4<f32>(arow1.y) * brow + sum01;\n    sum02 = vec4<f32>(arow2.y) * brow + sum02;\n    sum03 = vec4<f32>(arow3.y) * brow + sum03;\n    brow = array_b[(k * 4u + 1u) * ND4 + x * 2u + 1u];\n    sum10 = vec4<f32>(arow0.y) * brow + sum10;\n    sum11 = vec4<f32>(arow1.y) * brow + sum11;\n    sum12 = vec4<f32>(arow2.y) * brow + sum12;\n    sum13 = vec4<f32>(arow3.y) * brow + sum13;\n    \n    brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 0u];\n    sum00 = vec4<f32>(arow0.z) * brow + sum00;\n    sum01 = vec4<f32>(arow1.z) * brow + sum01;\n    sum02 = vec4<f32>(arow2.z) * brow + sum02;\n    sum03 = vec4<f32>(arow3.z) * brow + sum03;\n    brow = array_b[(k * 4u + 2u) * ND4 + x * 2u + 1u];\n    sum10 = vec4<f32>(arow0.z) * brow + sum10;\n    sum11 = vec4<f32>(arow1.z) * brow + sum11;\n    sum12 = vec4<f32>(arow2.z) * brow + sum12;\n    sum13 = vec4<f32>(arow3.z) * brow + sum13;\n    \n    brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 0u];\n    sum00 = vec4<f32>(arow0.w) * brow + sum00;\n    sum01 = vec4<f32>(arow1.w) * brow + sum01;\n    sum02 = vec4<f32>(arow2.w) * brow + sum02;\n    sum03 = vec4<f32>(arow3.w) * brow + sum03;\n    brow = array_b[(k * 4u + 3u) * ND4 + x * 2u + 1u];\n    sum10 = vec4<f32>(arow0.w) * brow + sum10;\n    sum11 = vec4<f32>(arow1.w) * brow + sum11;\n    sum12 = vec4<f32>(arow2.w) * brow + sum12;\n    sum13 = vec4<f32>(arow3.w) * brow + sum13;\n  }\n  array_c[x * 2u + 0u + (y * 4u + 0u) * ND4] = sum00 * alpha;\n  array_c[x * 2u + 0u + (y * 4u + 1u) * ND4] = sum01 * alpha;\n  array_c[x * 2u + 0u + (y * 4u + 2u) * ND4] = sum02 * alpha;\n  array_c[x * 2u + 0u + (y * 4u + 3u) * ND4] = sum03 * alpha;\n  array_c[x * 2u + 1u + (y * 4u + 0u) * ND4] = sum10 * alpha;\n  array_c[x * 2u + 1u + (y * 4u + 1u) * ND4] = sum11 * alpha;\n  array_c[x * 2u + 1u + (y * 4u + 2u) * ND4] = sum12 * alpha;\n  array_c[x * 2u + 1u + (y * 4u + 3u) * ND4] = sum13 * alpha;\n}\n`;\n\n\n//# sourceURL=webpack:///./node_modules/webgpu-blas/dist/shader_sgemm_block.js?");

/***/ }),

/***/ "./node_modules/webgpu-blas/dist/shader_sgemm_generic.js":
/*!***************************************************************!*\
  !*** ./node_modules/webgpu-blas/dist/shader_sgemm_generic.js ***!
  \***************************************************************/
/*! no static exports found */
/***/ (function(module, exports, __webpack_require__) {

"use strict";
eval("\nObject.defineProperty(exports, \"__esModule\", { value: true });\nexports.Shader = void 0;\nexports.Shader = `\n@group(0) @binding(0)\nvar<storage,read> array_a: array<f32>;\n\n@group(0) @binding(1)\nvar<storage,read> array_b: array<f32>;\n\n@group(0) @binding(2)\nvar<storage,read_write> array_c: array<f32>;\n\nstruct Meta {\n  M: f32;\n  N: f32;\n  K: f32;\n  alpha: f32;\n}\n\n@group(0) @binding(3)\nvar<storage,read> meta: Meta;\n\n@stage(compute) @workgroup_size(8,8,1)\nfn main(\n  @builtin(global_invocation_id) global_id: vec3<u32>\n) {\n  var M: u32 = u32(meta.M);\n  var N: u32 = u32(meta.N);\n  var K: u32 = u32(meta.K);\n  var x: u32 = global_id.x;\n  var y: u32 = global_id.y;\n  if (x >= N || y >= M) {\n    return;\n  }\n  var sum: f32 = 0.0;\n  for(var k: u32 = 0u; k < K; k = k + 1u) {\n    sum = array_a[y * K + k] * array_b[k * N + x] + sum;\n  }\n  array_c[x + y * N] = sum * meta.alpha;\n}\n`;\n\n\n//# sourceURL=webpack:///./node_modules/webgpu-blas/dist/shader_sgemm_generic.js?");

/***/ }),

/***/ "./src/main.ts":
/*!*********************!*\
  !*** ./src/main.ts ***!
  \*********************/
/*! no exports provided */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony import */ var webgpu_blas__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! webgpu-blas */ \"./node_modules/webgpu-blas/dist/index.js\");\n/* harmony import */ var webgpu_blas__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(webgpu_blas__WEBPACK_IMPORTED_MODULE_0__);\nvar __awaiter = (undefined && undefined.__awaiter) || function (thisArg, _arguments, P, generator) {\r\n    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }\r\n    return new (P || (P = Promise))(function (resolve, reject) {\r\n        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }\r\n        function rejected(value) { try { step(generator[\"throw\"](value)); } catch (e) { reject(e); } }\r\n        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }\r\n        step((generator = generator.apply(thisArg, _arguments || [])).next());\r\n    });\r\n};\r\nvar __generator = (undefined && undefined.__generator) || function (thisArg, body) {\r\n    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;\r\n    return g = { next: verb(0), \"throw\": verb(1), \"return\": verb(2) }, typeof Symbol === \"function\" && (g[Symbol.iterator] = function() { return this; }), g;\r\n    function verb(n) { return function (v) { return step([n, v]); }; }\r\n    function step(op) {\r\n        if (f) throw new TypeError(\"Generator is already executing.\");\r\n        while (_) try {\r\n            if (f = 1, y && (t = op[0] & 2 ? y[\"return\"] : op[0] ? y[\"throw\"] || ((t = y[\"return\"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;\r\n            if (y = 0, t) op = [op[0] & 2, t.value];\r\n            switch (op[0]) {\r\n                case 0: case 1: t = op; break;\r\n                case 4: _.label++; return { value: op[1], done: false };\r\n                case 5: _.label++; y = op[1]; op = [0]; continue;\r\n                case 7: op = _.ops.pop(); _.trys.pop(); continue;\r\n                default:\r\n                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }\r\n                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }\r\n                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }\r\n                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }\r\n                    if (t[2]) _.ops.pop();\r\n                    _.trys.pop(); continue;\r\n            }\r\n            op = body.call(thisArg, _);\r\n        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }\r\n        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };\r\n    }\r\n};\r\n\r\nfunction message(m, target) {\r\n    document.getElementById(target).innerText += m + '\\n';\r\n}\r\nfunction makeRandom(length) {\r\n    var array = new Float32Array(length);\r\n    for (var i = 0; i < length; i++) {\r\n        array[i] = Math.random();\r\n    }\r\n    return array;\r\n}\r\nfunction checkResult(m, n, k, alpha, array_a, array_b, actual) {\r\n    var expected = new Float32Array(m * n);\r\n    for (var row = 0; row < m; row++) {\r\n        for (var col = 0; col < n; col++) {\r\n            var sum = 0.0;\r\n            for (var j = 0; j < k; j++) {\r\n                sum += array_a[row * k + j] * array_b[j * n + col];\r\n            }\r\n            expected[row * n + col] = sum * alpha;\r\n        }\r\n    }\r\n    for (var row = 0; row < m; row++) {\r\n        for (var col = 0; col < n; col++) {\r\n            var idx = row * n + col;\r\n            var expected_el = expected[idx];\r\n            var actual_el = actual[idx];\r\n            if (Math.abs(expected_el - actual_el) > (1e-5 + 1e-3 * Math.abs(expected_el))) {\r\n                console.error(\"[\" + row + \", \" + col + \"]: \" + expected_el + \" !== \" + actual_el);\r\n                return false;\r\n            }\r\n        }\r\n    }\r\n    return true;\r\n}\r\nfunction parseMNKTuples(s) {\r\n    var shapes = []; //[[m,n,k]]\r\n    for (var _i = 0, _a = s.split('\\n'); _i < _a.length; _i++) {\r\n        var line = _a[_i];\r\n        var parts = line.split(',').map(function (t) { return Number(t.trim()); });\r\n        if (parts.length === 3 && parts.every(function (v) { return v > 0; })) {\r\n            shapes.push(parts);\r\n        }\r\n    }\r\n    return shapes;\r\n}\r\nfunction run_benchmark() {\r\n    return __awaiter(this, void 0, void 0, function () {\r\n        var messageTarget, shapes, alpha, runs, _i, shapes_1, _a, m, n, k, array_a, array_b, timeSum, retSum, i, sgemmStartTime, result, sgemmEndTime, avgTime, flops, ex_1;\r\n        return __generator(this, function (_b) {\r\n            switch (_b.label) {\r\n                case 0:\r\n                    messageTarget = 'bench_message';\r\n                    _b.label = 1;\r\n                case 1:\r\n                    _b.trys.push([1, 10, , 11]);\r\n                    shapes = parseMNKTuples(document.getElementById('benchmark_shapes').value);\r\n                    alpha = 1.0;\r\n                    runs = 10;\r\n                    _i = 0, shapes_1 = shapes;\r\n                    _b.label = 2;\r\n                case 2:\r\n                    if (!(_i < shapes_1.length)) return [3 /*break*/, 9];\r\n                    _a = shapes_1[_i], m = _a[0], n = _a[1], k = _a[2];\r\n                    array_a = makeRandom(m * k);\r\n                    array_b = makeRandom(k * n);\r\n                    // warmup\r\n                    return [4 /*yield*/, Object(webgpu_blas__WEBPACK_IMPORTED_MODULE_0__[\"sgemm\"])(m, n, k, alpha, array_a, array_b)];\r\n                case 3:\r\n                    // warmup\r\n                    _b.sent();\r\n                    timeSum = 0;\r\n                    retSum = 0;\r\n                    i = 0;\r\n                    _b.label = 4;\r\n                case 4:\r\n                    if (!(i < runs)) return [3 /*break*/, 7];\r\n                    console.time('sgemm');\r\n                    sgemmStartTime = performance.now();\r\n                    return [4 /*yield*/, Object(webgpu_blas__WEBPACK_IMPORTED_MODULE_0__[\"sgemm\"])(m, n, k, alpha, array_a, array_b)];\r\n                case 5:\r\n                    result = _b.sent();\r\n                    retSum += result[0];\r\n                    sgemmEndTime = performance.now();\r\n                    console.timeEnd('sgemm');\r\n                    timeSum += sgemmEndTime - sgemmStartTime;\r\n                    _b.label = 6;\r\n                case 6:\r\n                    i++;\r\n                    return [3 /*break*/, 4];\r\n                case 7:\r\n                    avgTime = timeSum / runs;\r\n                    flops = m * n * k * 2 * 1000 / avgTime / 1000000000;\r\n                    message(\"Sgemm of (\" + m + \"x\" + k + \"),(\" + k + \"x\" + n + \"): average \" + avgTime + \" ms (\" + runs + \" runs), \" + flops.toFixed(2) + \" GFLOPS\", messageTarget);\r\n                    console.log('sum of result (to avoid optimization)', retSum);\r\n                    _b.label = 8;\r\n                case 8:\r\n                    _i++;\r\n                    return [3 /*break*/, 2];\r\n                case 9: return [3 /*break*/, 11];\r\n                case 10:\r\n                    ex_1 = _b.sent();\r\n                    alert(ex_1.message);\r\n                    return [3 /*break*/, 11];\r\n                case 11: return [2 /*return*/];\r\n            }\r\n        });\r\n    });\r\n}\r\nfunction small_example() {\r\n    return __awaiter(this, void 0, void 0, function () {\r\n        var array_a, array_b, result, ex_2;\r\n        return __generator(this, function (_a) {\r\n            switch (_a.label) {\r\n                case 0:\r\n                    _a.trys.push([0, 2, , 3]);\r\n                    array_a = new Float32Array([1, 2, 3, 4]);\r\n                    array_b = new Float32Array([5, 6, 7, 8]);\r\n                    return [4 /*yield*/, Object(webgpu_blas__WEBPACK_IMPORTED_MODULE_0__[\"sgemm\"])(2, 2, 2, 1, array_a, array_b)];\r\n                case 1:\r\n                    result = _a.sent();\r\n                    document.getElementById('small_example_result').innerText = \"[\" + result[0] + \", \" + result[1] + \"\\n \" + result[2] + \", \" + result[3] + \"]\";\r\n                    return [3 /*break*/, 3];\r\n                case 2:\r\n                    ex_2 = _a.sent();\r\n                    alert(ex_2.message);\r\n                    return [3 /*break*/, 3];\r\n                case 3: return [2 /*return*/];\r\n            }\r\n        });\r\n    });\r\n}\r\nfunction run_test() {\r\n    return __awaiter(this, void 0, void 0, function () {\r\n        var shapes, alpha, messageTarget, _i, shapes_2, _a, m, n, k, array_a, array_b, result, validation_result;\r\n        return __generator(this, function (_b) {\r\n            switch (_b.label) {\r\n                case 0:\r\n                    shapes = parseMNKTuples(document.getElementById('test_shapes').value);\r\n                    alpha = 1.0;\r\n                    messageTarget = 'test_message';\r\n                    _i = 0, shapes_2 = shapes;\r\n                    _b.label = 1;\r\n                case 1:\r\n                    if (!(_i < shapes_2.length)) return [3 /*break*/, 4];\r\n                    _a = shapes_2[_i], m = _a[0], n = _a[1], k = _a[2];\r\n                    array_a = makeRandom(m * k);\r\n                    array_b = makeRandom(k * n);\r\n                    return [4 /*yield*/, Object(webgpu_blas__WEBPACK_IMPORTED_MODULE_0__[\"sgemm\"])(m, n, k, alpha, array_a, array_b)];\r\n                case 2:\r\n                    result = _b.sent();\r\n                    validation_result = checkResult(m, n, k, alpha, array_a, array_b, result);\r\n                    message(\"M=\" + m + \", N=\" + n + \", K=\" + k + \": \" + (validation_result ? 'OK' : 'Error'), messageTarget);\r\n                    _b.label = 3;\r\n                case 3:\r\n                    _i++;\r\n                    return [3 /*break*/, 1];\r\n                case 4: return [2 /*return*/];\r\n            }\r\n        });\r\n    });\r\n}\r\nwindow.addEventListener('load', function () {\r\n    document.getElementById('run_benchmark').onclick = run_benchmark;\r\n    document.getElementById('small_example').onclick = small_example;\r\n    document.getElementById('run_test').onclick = run_test;\r\n    document.getElementById('is_webgpu_enabled').innerText = navigator.gpu ? 'Enabled' : 'Disabled (fallback pure JavaScript implementation will be used)';\r\n});\r\n\n\n//# sourceURL=webpack:///./src/main.ts?");

/***/ })

/******/ });