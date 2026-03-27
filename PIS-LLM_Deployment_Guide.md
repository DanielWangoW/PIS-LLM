# PIS-LLM 项目完整上传与 Streamlit Cloud 部署操作手册

这份指南将带您完成从本地代码提交、推送到您刚创建的 GitHub 仓库 (`git@github.com:DanielWangoW/PIS-LLM.git`)，直到在 Streamlit Community Cloud 上线公网访问的全流程。

---

## 第一阶段：解决 GitHub SSH 访问权限问题

系统检测到您本地的 `ssh -T git@github.com` 返回了 `Permission denied (publickey)`。这意味着您的本地 SSH 公钥尚未添加到 GitHub 账户中。在推送代码前，必须先完成此配置。

### 1. 复制本地公钥
在您的 Mac 终端（Terminal）中运行以下命令，将公钥复制到剪贴板：
```bash
pbcopy < ~/.ssh/id_rsa.pub
```
*(注：如果上述命令报错找不到文件，说明您尚未生成过 SSH 密钥，请先运行 `ssh-keygen -t rsa -b 4096 -C "daomiao.wang@live.com"` 一路回车生成，然后再执行 `pbcopy`。)*

### 2. 添加到 GitHub
1. 登录 GitHub，点击右上角头像，选择 **Settings**。
2. 在左侧菜单中选择 **SSH and GPG keys**。
3. 点击绿色的 **New SSH key** 按钮。
4. **Title** 填入 `MacBook Pro`（或任意易记名称）。
5. **Key type** 保持默认的 `Authentication Key`。
6. 在 **Key** 输入框中，按 `Cmd + V` 粘贴刚才复制的公钥内容。
7. 点击 **Add SSH key**。

### 3. 测试连接
回到 Mac 终端，再次运行：
```bash
ssh -T git@github.com
```
如果看到类似 `Hi DanielWangoW! You've successfully authenticated...` 的提示，说明配置成功。

---

## 第二阶段：提交代码并推送到 GitHub

您的本地项目目录已经配置好了极其严格的 `.gitignore`，所有敏感文件（如 `.env`, `apikey.txt`）和大型本地生成文件（如 `outputs/`, `images/`）都已被自动排除，非常安全。

请在 Mac 终端中依次执行以下命令：

### 1. 进入项目目录并提交代码
```bash
cd /Users/danielwang/Documents/CodeSpace/WS_Jupyter/ProEngOpt

# 将所有未被 .gitignore 排除的文件添加到暂存区
git add .

# 提交代码
git commit -m "feat: Initial commit of PIS-LLM (CardioDetector TDA v2.1)"
```

### 2. 关联远程仓库并推送
```bash
# 关联您刚才创建的 GitHub 仓库
git remote add origin git@github.com:DanielWangoW/PIS-LLM.git

# 确保当前分支名为 main
git branch -M main

# 推送代码到 GitHub
git push -u origin main
```
推送完成后，刷新您的 GitHub 仓库页面，应该能看到包含 `README.md`、`main.py` 等文件的完整代码树。

---

## 第三阶段：在 Streamlit Cloud 上部署应用

Streamlit Community Cloud 提供免费的公网托管服务，并能直接从您的 GitHub 仓库拉取代码。我们已经为您准备好了云端所需的系统依赖文件 `packages.txt` 和 Python 依赖文件 `requirements.txt`。

### 1. 创建 Streamlit 应用
1. 访问 [share.streamlit.io](https://share.streamlit.io) 并使用您的 GitHub 账号登录。
2. 点击右上角的 **Create app** 按钮。
3. 选择 **Yep, I have an app**。
4. 填写部署信息：
   - **Repository**: 选择 `DanielWangoW/PIS-LLM`
   - **Branch**: `main`
   - **Main file path**: 填入 `main.py`
   - **App URL**: 可以自定义一个专属后缀，例如 `pis-llm`

### 2. 配置 API Keys (Secrets)
由于代码中没有包含真实的 API Key（这非常安全），我们需要在云端控制台将 Key 注入给应用。

1. 在点击 "Deploy" 之前，点击页面底部的 **Advanced settings...**。
2. 找到 **Secrets** 输入框。
3. 按照 TOML 格式，填入您的 LLM API Keys。您可以参考项目中的 `.streamlit/secrets.toml.example` 格式：

```toml
[api_keys]
DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
ZHIPU_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxx"
MOONSHOT_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```
*(注：只需填入您打算使用的模型的 Key 即可，不用的可以留空或不写)*

4. 点击 **Save**，然后点击 **Deploy!**。

### 3. 等待构建完成
Streamlit Cloud 会自动读取 `packages.txt` 安装 C++ 依赖（`gudhi` 所需），然后读取 `requirements.txt` 安装 Python 库。首次构建大约需要 2-3 分钟。
构建完成后，应用会自动启动，您将获得一个类似 `https://pis-llm.streamlit.app` 的公网链接。

---

## 常见问题排查 (Troubleshooting)

- **Q: 部署时报错 `ModuleNotFoundError: No module named 'gudhi'`**
  - **A**: 请检查 GitHub 仓库根目录下是否存在 `packages.txt` 且内容包含 `libboost-dev` 等依赖。如果没有，请确保将其 push 到仓库。

- **Q: 网页显示出来了，但点击生成报告时报错 `API Key missing`**
  - **A**: 说明 Streamlit Cloud 的 Secrets 没有正确读取。请点击应用右下角的 **Manage app** -> 三个点菜单 -> **Settings** -> **Secrets**，检查 `[api_keys]` 的格式是否正确（必须是 TOML 格式的键值对，带双引号）。

- **Q: 应用长时间不用后访问很慢**
  - **A**: Streamlit Community Cloud 会在应用闲置 7 天后将其休眠。再次访问时会有几十秒的 "Waking up" 冷启动时间，这是正常现象。您也可以在控制台手动唤醒它。
