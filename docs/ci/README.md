# 自动测试配置

`tests.yml.example` 是已准备好的 GitHub Actions 工作流模板，当前尚未启用。

首次上传使用的 GitHub OAuth 凭据具有 `repo` 权限但缺少 `workflow` 权限；GitHub 因而拒绝包含 `.github/workflows/tests.yml` 的推送。代码上传与问题跟踪不依赖此权限。

后续由仓库所有者为 GitHub CLI 授予 `workflow` 权限（`gh auth refresh -h github.com -s workflow`），再将模板复制为 `.github/workflows/tests.yml` 并完成一次配置提交，即可启用。

模板使用 Windows + Python 3.12，运行 pytest 和 Qt offscreen 界面测试；没有私有样本时对应测试跳过，不运行原生桌面操作工具。权限仅为 `contents: read`。

本次公开发布前，本地完整测试 234 项通过；不含私有数据的干净副本中 231 项通过、3 项私有样本测试跳过。
