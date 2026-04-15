# T5 Text Summarizer 🚀

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A GPU-accelerated text summarization web application powered by a fine-tuned T5 (Text-to-Text Transfer Transformer) model, served via a FastAPI backend with a premium, dark-themed UI.**

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Project Structure](#-project-structure)

</div>

---

## ✨ Features

- 🤖 **Fine-tuned T5 Model** — Custom-trained T5-base model (`t5-base`) optimized for abstractive text summarization
- ⚡ **GPU Acceleration** — Enforced CUDA inference via PyTorch for fast, real-time summaries
- 🌐 **FastAPI Backend** — High-performance async REST API with automatic OpenAPI docs
- 🎨 **Premium Web UI** — Dark-themed, glassmorphism interface with Three.js animated shader background
- 📏 **Adjustable Length** — Real-time slider to control `max_token_length` (50–500 tokens)
- 📋 **One-click Copy** — Instantly copy the generated summary to clipboard
- 🧹 **Text Preprocessing** — Removes HTML tags, extra whitespace, and newlines before inference

---

## 🏗️ Architecture

```
User Browser
    │
    ▼
┌─────────────────────────────────────┐
│         FastAPI Application          │
│  ┌──────────────────────────────┐   │
│  │  POST /api/summarize         │   │
│  │  ├─ Input cleaning           │   │
│  │  ├─ Tokenization (seq ≤1024) │   │
│  │  └─ Beam Search (n=4)        │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │  GET /  → static/index.html  │   │
│  └──────────────────────────────┘   │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  T5ForConditional    │
    │  Generation (CUDA)   │
    │  saved_summarizer_   │
    │  model/              │
    └──────────────────────┘
```

**Model Specs:**
| Parameter | Value |
|---|---|
| Architecture | T5 Encoder-Decoder |
| Model Dimension (`d_model`) | 768 |
| Feed-Forward Dimension (`d_ff`) | 3072 |
| Attention Heads | 12 |
| Encoder Layers | 12 |
| Decoder Layers | 12 |
| Vocab Size | 32,128 |
| Input Max Length | 1024 tokens |
| Output Length (default) | 40–150 tokens |
| Decoding Strategy | Beam Search (4 beams, length_penalty=2.0) |

---

## 📁 Project Structure

```
T5_Text_Summarizer/
│
├── main.py                        # FastAPI app — API routes & model loading
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules (excludes large model weights)
├── LICENSE                        # MIT License
├── README.md                      # Project documentation (this file)
│
├── static/                        # Frontend web app
│   ├── index.html                 # Main page — dual-panel UI (input / output)
│   ├── style.css                  # Dark glassmorphism CSS (Inter + JetBrains Mono)
│   └── script.js                  # API calls, slider logic, Three.js shader setup
│
└── saved_summarizer_model/        # Fine-tuned T5 model artifacts
    ├── config.json                # Model architecture config
    ├── generation_config.json     # Beam search & generation defaults
    ├── model.safetensors          # Model weights (~850 MB) — excluded from Git
    ├── tokenizer.json             # SentencePiece vocabulary & rules
    └── tokenizer_config.json      # Tokenizer settings & special tokens
```

> **Note:** `model.safetensors` (~850 MB) is excluded from this repository via `.gitignore`. Download or train the model locally (see [Training](#-training-the-model)).

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA drivers installed *(required — CPU inference is disabled)*
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/nisargpatel1906/T5_Text_Summarizer.git
cd T5_Text_Summarizer
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> For GPU support, install the CUDA-compatible version of PyTorch:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Place the Trained Model

Ensure your fine-tuned model files are available at:

```
saved_summarizer_model/
├── config.json
├── generation_config.json
├── model.safetensors      ← required (not in Git)
├── tokenizer.json
└── tokenizer_config.json
```

### 5. Run the Application

```bash
python main.py
```

The server will start at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

Open the URL in your browser to access the web interface.

---

## 🛠️ API Reference

### `POST /api/summarize`

Summarizes the provided input text using the fine-tuned T5 model.

**Request Body:**

```json
{
  "text": "Your article or document text here...",
  "max_length": 150,
  "min_length": 40
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `string` | — | The input text to summarize *(required)* |
| `max_length` | `integer` | `150` | Maximum token length of the generated summary |
| `min_length` | `integer` | `40` | Minimum token length of the generated summary |

**Success Response (`200 OK`):**

```json
{
  "summary": "A concise and coherent summary of the input text."
}
```

**Error Responses:**

| Status | Reason |
|---|---|
| `400` | Input text is empty |
| `500` | Model not loaded or inference error |

### `GET /`

Serves the main `index.html` web interface.

### Interactive Docs

FastAPI auto-generates interactive API documentation:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🧪 Training the Model

The model was fine-tuned using the Hugging Face `Trainer` API on a summarization dataset. The training notebook is included in the repository:

📓 **`text_summarizer.ipynb`** — Complete training pipeline including:
- Dataset loading and preprocessing
- T5 tokenization with `summarize:` prefix
- `TrainingArguments` configuration
- Evaluation using ROUGE metrics
- Model checkpointing and saving

**Key Training Details:**
- Base model: `t5-base`
- Task prefix: `"summarize: "`
- Input truncation: 1024 tokens
- Target truncation: 128 tokens
- Decoding: Beam search with `num_beams=4`, `length_penalty=2.0`, `early_stopping=True`

> ⚠️ **Checkpoint Note:** The model used in this application is taken from **checkpoint-4000**.
> Training was stopped at step 4000 because the model began **overfitting** beyond that point — validation loss started increasing while training loss kept decreasing.
> Using checkpoint-4000 gives the best generalization and summary quality on unseen text.

---

## 🖥️ Web Interface

The frontend is a single-page application with:

- **Dual-panel layout** — Source text input (left) and AI output (right)
- **Three.js animated background** — WebGL shader-based dark animated background
- **Glassmorphism panels** — Frosted-glass card design with subtle glow effects
- **Word count indicator** — Live word count as you type
- **Loading skeleton** — Shimmer animation while awaiting inference
- **Copy to clipboard** — One-click copy button on generated summary
- **Fonts:** [Inter](https://fonts.google.com/specimen/Inter) + [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono) from Google Fonts

---

## ⚙️ Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework for the REST API |
| `uvicorn` | ASGI server for running FastAPI |
| `pydantic` | Request/response data validation |
| `transformers` | T5 model & tokenizer (Hugging Face) |
| `torch` | PyTorch for GPU inference |

---

## 🔭 Future Goals

The following improvements are planned for upcoming versions of this project:

| # | Goal | Description |
|---|---|---|
| 1 | 🧠 **Upgrade to T5-Large / FLAN-T5** | Replace `t5-base` with `t5-large` or Google's `flan-t5-large` for better factual accuracy and fluency in summaries |
| 2 | 📂 **Multi-File Batch Summarization** | Allow users to upload multiple `.txt` or `.pdf` files at once and download a combined summary report |
| 3 | 🌍 **Multi-Language Support** | Add support for summarizing text in multiple languages (Hindi, French, Spanish) using multilingual T5 (`mT5`) |
| 4 | 📊 **ROUGE Score Display** | Show real-time ROUGE-1, ROUGE-2, and ROUGE-L scores alongside each generated summary so users can gauge quality |
| 5 | 🗂️ **Summary History & Export** | Save past summaries in a local session log with the ability to export them as `.txt` or `.pdf` files |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork this repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'feat: add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.

---

## 👤 Author

**Nisarg Patel**

- GitHub: [@nisargpatel1906](https://github.com/nisargpatel1906)

---

<div align="center">
  <sub>Built with ❤️ using T5, FastAPI & PyTorch · Fine-tuned for abstractive text summarization</sub>
</div>
