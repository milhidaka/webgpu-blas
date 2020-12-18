export default function glslangModule(): Promise<GLSLang>;

export class GLSLang {
    compileGLSL(shaderCode: string, type: 'compute'): Uint32Array;
}
