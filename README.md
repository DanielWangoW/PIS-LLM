# PIS-LLM: Topological Data Analysis and Large Language Models for Cardiovascular Signal Assessment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PIS-LLM** (Persistent Homology-Informed Signal Large Language Model) is a novel computational framework that integrates **Topological Data Analysis (TDA)** with **Large Language Models (LLMs)** for automated, explainable cardiovascular signal assessment. 

This repository contains the official implementation of the PIS-LLM framework, including the TDA signal processing pipeline, cardiovascular metrics extraction, and the interactive web-based clinical report generator.

---

## 🌟 Key Features

- **Topological Signal Processing**: Transforms 1D cardiovascular signals (e.g., PPG, ECG) into high-dimensional phase spaces using Takens' delay embedding.
- **Robust Anomaly Detection**: Utilizes Weighted Rips Complexes and persistent homology (via `gudhi`) to capture structural topological cycles, effectively distinguishing between normal physiological rhythms and arrhythmic/anomalous events.
- **Multi-Model LLM Integration**: Generates structured, factual, and strictly quantitative-based clinical assessment reports. Supports leading models including **Qwen, DeepSeek, GLM, and Kimi** via OpenAI-compatible APIs.
- **Explainable AI in Healthcare**: Bridges the gap between complex mathematical topology and interpretable clinical insights without fabricating patient histories or identities.
- **Modern Interactive UI**: A dark-themed, glassmorphism Streamlit web application providing end-to-end signal upload, real-time TDA visualization, and conversational follow-ups based on the generated reports.

---

## 🏗️ System Architecture

The PIS-LLM framework operates through a decoupled, three-stage pipeline:

1. **Signal Processing & TDA Module (`core/tda_lib/TDA_4_1DTS.py`)**
   - Ingests raw time-series data.
   - Constructs Takens delay embeddings.
   - Computes persistent homology to extract 1-dimensional topological cycles ($H_1$).
   - Calculates point-to-cycle distances to generate a continuous anomaly profile.

2. **Cardiovascular Metrics Extractor (`core/tda_lib/CardiovascularMetrics.py`)**
   - Quantifies topological features (e.g., mean persistence, anomaly ratio).
   - Estimates physiological metrics (e.g., heart rate, dominant frequency).
   - Categorizes anomaly severity (Mild, Moderate, Severe).

3. **LLM Report Generator (`core/llm_client.py` & `prompts/`)**
   - Injects the quantitative metrics into heavily constrained prompt templates (available in English and Chinese).
   - Streams a structured cardiovascular health assessment report.
   - Maintains conversation history for multi-turn clinical Q&A based strictly on the TDA results.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda (Recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/danielwangow/ProEngOpt.git
cd ProEngOpt
```

### 2. Create Environment and Install Dependencies
```bash
conda create -n pis-llm python=3.11 -y
conda activate pis-llm

# Install core dependencies (including gudhi for TDA)
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy the environment template and add your preferred LLM provider API keys:
```bash
cp .env.example .env
```
Edit `.env` to include your keys (e.g., `DASHSCOPE_API_KEY`, `DEEPSEEK_API_KEY`). Alternatively, you can input the API key directly in the web UI.

---

## 💻 Usage

Start the interactive Streamlit application:

```bash
conda activate pis-llm
streamlit run main.py
```

1. **Upload Signal**: Upload a `.csv` or `.txt` file containing 1D cardiovascular signal data.
2. **Adjust TDA Parameters**: (Optional) Fine-tune Takens embedding dimension ($d$), time delay ($\tau$), and persistence thresholds via the sidebar.
3. **Run Analysis**: Click "Run Full Analysis" to execute the TDA pipeline.
4. **View Results**: Explore the generated phase space projections, persistence barcodes, and the LLM-generated clinical report.
5. **Chat**: Use the chat interface at the bottom to ask the LLM specific questions about the signal's topological features.

---

## 📄 Citation

If you find this code or the PIS-LLM framework useful in your research, please cite our paper:

<!-- ```bibtex
@article{wang2026pisllm,
  title={PIS-LLM: Persistent Homology-Informed Signal Large Language Models for Cardiovascular Assessment},
  author={Wang, Daniel and others},
  journal={Submit to [Journal/Conference Name]},
  year={2026},
  institution={Fudan University}
}
``` -->
*(Note: Citation details will be updated upon publication.)*

---

## ⚠️ Disclaimer

**PIS-LLM is a computational screening and research tool.** The generated reports and topological analyses are for academic and research purposes only and **do not constitute medical diagnoses**. Always consult a qualified healthcare professional for medical advice.

---
**Author:** Daniel Wang @ Fudan University  
**Contact:** daomiao.wang@live.com  
**Copyright:** © 2026 Fudan University. All Rights Reserved.
