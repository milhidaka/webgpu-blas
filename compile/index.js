import glslangModule from "https://unpkg.com/@webgpu/glslang@0.0.8/dist/web-devel/glslang.js";

let glslang

async function compile() {
    if (!glslang) {
        glslang = await glslangModule();
    }
    const source = document.getElementById('source').value;
    
    const glslShader = glslang.compileGLSL(source, "compute");
    const compiled = `export const Shader = new Uint32Array([${Array.from(glslShader).toString()}]);`
    document.getElementById('compiled').value = compiled;
}

window.addEventListener('DOMContentLoaded', () => {
    document.getElementById('run_compile').onclick = compile;
});
