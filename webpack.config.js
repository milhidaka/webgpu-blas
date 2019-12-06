module.exports = {
  mode: 'development',
  entry: './src/index.ts',

  output: {
    filename: 'webgpublas.js',
    path: __dirname + '/dist',
    library: 'webgpublas',
    libraryTarget: 'var'
  },

  module: {
    rules: [
      {
        test: /\.ts$/,
        use: 'ts-loader'
      }
    ]
  },
  resolve: {
    extensions: [
      '.ts',
      '.js'
    ]
  }
};