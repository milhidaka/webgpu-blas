# 各種リリース手段

バージョンアップしたらリリースする。

# 準備
`package.json` でバージョン番号を更新してcommit・push。

# Github release

Github上のreleaseを行う。 `dist/webgpublas.js` を添付する。

# Github pages

`examples/sgemm/dist` 以下を `gh-pages` ブランチのルートにコピー。

# npm

```
npm publish
```
