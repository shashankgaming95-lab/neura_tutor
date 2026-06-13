# 🧠 NeuraTutor — Emotion-Aware Virtual Learning Assistant

> NeuroneX Hackathon Submission · No API key required · Runs 100% locally or on Kaggle

NeuraTutor is an AI tutor that detects how you're feeling and **completely changes** how it teaches — simpler when you're confused, more challenging when you're confident, exciting when you're bored.

---

## How it works

```
Your mood  →  Flask backend  →  HuggingFace LLM  →  Adapted response
(confused)     (app.py)          (TinyLlama)         (step-by-step, warm tone)
```

The model runs entirely on your machine or Kaggle — no internet, no API key, no cost.

---

## Project structure

```
neura-tutor/
├── app.py                  # Flask backend + HuggingFace model
├── templates/
│   └── index.html          # Chat UI (served by Flask)
├── requirements.txt        # Python dependencies
├── kaggle_notebook.ipynb   # Run on Kaggle + expose via ngrok
└── README.md
```

---

## Run locally on your computer

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/neura-tutor.git
cd neura-tutor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the server

```bash
python app.py
```

The first run downloads the model (~600 MB). You'll see:

```
Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Model loaded! Server starting...
```

### 4. Open the app

Open your browser and go to:

```
http://localhost:5000
```

That's it. The app is fully running locally.

---

## Run on Kaggle (free GPU)

Kaggle gives you free T4 GPU which makes the model much faster.

### 1. Upload files to Kaggle

- Go to [kaggle.com](https://www.kaggle.com) → **Code** → **New Notebook**
- Upload `kaggle_notebook.ipynb`

### 2. Enable GPU

In the notebook sidebar: **Session options → Accelerator → GPU T4 x2**

### 3. Get a free ngrok token

- Sign up free at [dashboard.ngrok.com](https://dashboard.ngrok.com)
- Copy your auth token from the dashboard

### 4. Run the notebook

- Paste your ngrok token into the cell where it says `YOUR_NGROK_AUTH_TOKEN_HERE`
- Run all cells (Shift+Enter each one)
- After ~60 seconds you'll see a public URL like `https://xxxx.ngrok-free.app`

### 5. Open the URL in your browser

Paste the ngrok URL into any browser — the full NeuraTutor app opens. Share this link with anyone while your Kaggle session is running.

---

## What the AI does differently per emotion

| Your mood | Teaching style | How the AI responds |
|---|---|---|
| 😊 Confident | Concise | Advanced explanation, skips basics, harder follow-up |
| 🤔 Curious | Exploratory | Rich analogies, surprising facts, rabbit holes |
| 😕 Confused | Simplified | Tiny steps, no jargon, everyday analogies, warm tone |
| 😤 Frustrated | Supportive | Empathy first, fun metaphor, very short, encouraging |
| 😑 Bored | Engaging | Shocking fact opener, energetic, real-world story |

---

## Upgrade to a better model (optional)

In `app.py`, change line 16:

```python
# Default (fast, CPU-friendly, ~600MB)
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Better quality (needs GPU, ~14GB)
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
```

TinyLlama works well for demos. Mistral-7B gives much richer answers and is recommended on Kaggle GPU.

---

## Tech stack

| Layer | Technology |
|---|---|
| AI Model | TinyLlama 1.1B (HuggingFace Transformers) |
| Backend | Python + Flask |
| Frontend | Vanilla HTML/CSS/JS |
| Kaggle access | pyngrok tunnel |
| Cost | Free — no API key, no paid service |

---

## Team

**Squad:** hi · **Hackathon:** NeuroneX
