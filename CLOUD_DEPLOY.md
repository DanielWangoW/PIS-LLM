# CardioDetector TDA — Streamlit Community Cloud 部署指南

> 最小化上传文件集，一次 `git push` 完成公网部署。

---

## 第一步：确认上传文件清单

执行以下命令查看实际会被 git 追踪的文件（`.gitignore` 已过滤掉所有不必要内容）：

```bash
cd /Users/danielwang/Documents/CodeSpace/WS_Jupyter/ProEngOpt
git status --short
```

**应上传的文件（共约 15 个）：**

```
ProEngOpt/
├── main.py                        ← Streamlit 主程序
├── config.py                      ← 配置管理（含 Cloud secrets 读取）
├── requirements.txt               ← Python 依赖（已移除 dionysus）
├── packages.txt                   ← 系统依赖（gudhi 编译所需）
├── .env.example                   ← API Key 配置模板（无真实密钥）
├── .gitignore                     ← 排除规则
├── .streamlit/
│   └── secrets.toml.example       ← Cloud secrets 格式参考
├── core/
│   ├── __init__.py
│   ├── signal_processor.py
│   ├── llm_client.py
│   └── tda_lib/
│       ├── TDA_4_1DTS.py          ← 已移除 dionysus 依赖
│       ├── CardiovascularMetrics.py
│       └── dataPloter.py
└── prompts/
    ├── prompt_en.txt
    └── prompt_zh.txt
```

**不会上传（被 .gitignore 排除）：**

| 目录/文件 | 原因 |
|---|---|
| `.env` | 含真实 API Key，绝对不能上传 |
| `outputs/` | 运行时生成的图片（9.6 MB） |
| `images/` | 本地预览图（1.7 MB） |
| `upload_signals/` | 本地测试数据 |
| `_tda_tmp/` | 临时文件 |
| `__pycache__/` | Python 字节码 |
| `feedbacks/` | 本地反馈记录 |
| `parameters/results.json` | 本地运行参数缓存 |

---

## 第二步：初始化 Git 仓库并推送到 GitHub

```bash
cd /Users/danielwang/Documents/CodeSpace/WS_Jupyter/ProEngOpt

# 初始化（如果还没有 git 仓库）
git init
git add .
git status   # 确认只有上述 ~15 个文件被追踪

# 首次提交
git commit -m "feat: CardioDetector TDA v2.1 — cloud-ready, dionysus-free"

# 在 GitHub 上创建新的公开仓库（名称建议：ProEngOpt 或 cardiodetector-tda）
# 然后关联并推送：
git remote add origin https://github.com/danielwangow/ProEngOpt.git
git branch -M main
git push -u origin main
```

---

## 第三步：在 Streamlit Community Cloud 部署

1. 访问 [share.streamlit.io](https://share.streamlit.io) 并用 GitHub 账号登录
2. 点击右上角 **"Create app"**
3. 填写以下信息：

   | 字段 | 填写内容 |
   |---|---|
   | Repository | `danielwangow/ProEngOpt` |
   | Branch | `main` |
   | Main file path | `main.py` |
   | App URL (可选) | `cardiodetector-tda`（自定义子域名） |

4. 点击 **"Advanced settings"** → **"Secrets"**，粘贴以下内容（替换为真实 Key）：

   ```toml
   [api_keys]
   DASHSCOPE_API_KEY  = "sk-xxxxxxxxxxxxxxxx"
   DEEPSEEK_API_KEY   = "sk-xxxxxxxxxxxxxxxx"
   ZHIPU_API_KEY      = "xxxxxxxxxxxxxxxx"
   MOONSHOT_API_KEY   = "sk-xxxxxxxxxxxxxxxx"
   ```

5. 点击 **"Deploy!"**，等待约 3–5 分钟完成依赖安装

**部署完成后的公网地址：**
```
https://danielwangow-proengopt-main-xxxx.streamlit.app
```
或自定义子域名：
```
https://cardiodetector-tda.streamlit.app
```

---

## 第四步：更新展示页链接

将 GitHub Pages 展示页（`index.html`）中的 "Launch App" 按钮链接更新为实际地址：

```html
<!-- 找到所有 href="#deploy" 的 Launch App 按钮，替换为实际地址 -->
<a class="btn-primary" href="https://cardiodetector-tda.streamlit.app">🚀 Launch App</a>
```

然后推送展示页仓库：

```bash
cd /path/to/cardiodetector-tda-page
git add index.html
git commit -m "fix: update Launch App link to Streamlit Cloud URL"
git push
```

---

## 常见问题排查

**Q: 部署时 gudhi 安装失败**

确认 `packages.txt` 已在仓库根目录，内容为：
```
libgmp-dev
libmpfr-dev
libboost-dev
libboost-python-dev
```

**Q: 应用启动后报 `ModuleNotFoundError: No module named 'dionysus'`**

`TDA_4_1DTS.py` 已移除 dionysus 依赖。如果仍报错，说明推送的是旧版文件，请重新确认 `git status` 后再推送。

**Q: API Key 不生效**

检查 Streamlit Cloud Secrets 格式是否正确（必须是 TOML 格式，`[api_keys]` 节名称与 `config.py` 中一致）。

**Q: 应用休眠后访问慢**

免费版 Streamlit Cloud 在 30 分钟无访问后会休眠，首次唤醒约需 30 秒。如需 24/7 不休眠，可升级到 Streamlit Teams 版或改用云服务器方案。

---

## 后续更新流程

每次修改代码后，只需：

```bash
git add .
git commit -m "fix/feat: 描述改动"
git push
```

Streamlit Cloud 会自动检测到推送并重新部署，无需任何手动操作。
