"""
NeuraTutor — Emotion-Aware Virtual Learning Assistant
Backend: Flask + HuggingFace Transformers (no API key needed)
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

app = Flask(__name__)
CORS(app)

# ─── MODEL CONFIG ────────────────────────────────────────────────────────────
# TinyLlama is fast, free, runs on CPU — good for Kaggle free tier
# To upgrade: swap MODEL_ID for "mistralai/Mistral-7B-Instruct-v0.2" (needs GPU)
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"Loading model: {MODEL_ID}")
print("This may take a minute on first run (downloads ~600MB)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

print("Model loaded! Server starting...")

# ─── EMOTION PROFILES ────────────────────────────────────────────────────────
EMOTION_PROMPTS = {
    "confident": (
        "The student feels confident and capable. "
        "Give a concise, advanced explanation. Assume strong prior knowledge. "
        "Skip basics. Mention nuances or edge cases. "
        "End with one challenging follow-up question."
    ),
    "curious": (
        "The student is curious and engaged. "
        "Give a rich, exploratory explanation with interesting analogies and real-world examples. "
        "Mention a surprising fact. "
        "End with one thought-provoking question."
    ),
    "confused": (
        "The student is confused. "
        "Slow down. Use the simplest possible language. Break the concept into small numbered steps. "
        "Use a relatable everyday analogy. Avoid jargon entirely. Be warm and encouraging. "
        "End with one simple comprehension check."
    ),
    "frustrated": (
        "The student is frustrated. "
        "Start by acknowledging that this topic is genuinely hard. Be very warm and empathetic. "
        "Simplify drastically. Use a fun or surprising analogy. Keep the response short. "
        "End with an encouraging sentence."
    ),
    "bored": (
        "The student is bored. "
        "Make it exciting! Start with a surprising or counterintuitive fact. "
        "Use energetic language and add a real-world impact story. "
        "End with an intriguing question that sparks curiosity."
    ),
}

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "").strip()
    emotion = data.get("emotion", "curious").lower()

    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    if emotion not in EMOTION_PROMPTS:
        emotion = "curious"

    emotion_instruction = EMOTION_PROMPTS[emotion]

    # TinyLlama uses ChatML format
    system_prompt = (
        f"You are NeuraTutor, an emotion-aware AI learning assistant. "
        f"{emotion_instruction} "
        f"Keep your response under 150 words. "
        f"Bold key terms using **asterisks**. "
        f"Be direct and helpful."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    output = generator(
        prompt,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = output[0]["generated_text"]
    # Strip the prompt — return only the assistant reply
    reply = generated[len(prompt):].strip()

    # Convert **bold** markdown to <strong> HTML tags
    import re
    reply = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", reply)
    reply = reply.replace("\n", "<br>")

    return jsonify({"reply": reply, "emotion": emotion})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_ID})


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # In Kaggle use ngrok (see README). Locally just open localhost:5000
    app.run(host="0.0.0.0", port=port, debug=False)
