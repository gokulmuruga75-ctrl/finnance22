import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "chatbot_config.txt"
MODEL_NAME = "gemini-3.1-flash-lite"

app = Flask(__name__)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured in the .env file.")

client = genai.Client(api_key=api_key)

try:
    SYSTEM_PROMPT = CONFIG_FILE.read_text(encoding="utf-8").strip()
except OSError as exc:
    raise RuntimeError("chatbot_config.txt could not be loaded.") from exc


def is_study_question(message: str) -> bool:
    """Fast local guard for clearly unrelated requests."""
    study_terms = {
        "study", "learn", "learning", "education", "educational", "exam",
        "homework", "assignment", "lesson", "course", "class", "school",
        "college", "university", "student", "subject", "chapter", "notes",
        "explain", "concept", "definition", "formula", "solve", "problem",
        "question", "quiz", "revision", "research", "tutorial", "finance",
        "financial", "money", "budget", "saving", "investment", "investing",
        "stocks", "bond", "bonds", "mutual", "tax", "taxes", "banking",
        "bank", "loan", "credit", "interest", "inflation", "economics",
        "accounting", "business", "market", "portfolio", "risk", "insurance",
        "capital", "debt", "income", "expense", "profit", "loss", "economy",
    }

    words = set(message.lower().split())
    return bool(words & study_terms)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a question."}), 400

    # The model prompt remains the final authority for borderline cases.
    if not is_study_question(message):
        return jsonify({
            "answer": (
                "I’m FinanceGuide, a study-focused finance chatbot. "
                "Please ask me a finance or study-related question."
            )
        })

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=1200,
            ),
        )

        answer = (response.text or "").strip()
        if not answer:
            answer = "I couldn't generate an answer. Please try rephrasing your question."

        return jsonify({"answer": answer})

    except Exception:
        app.logger.exception("Gemini API request failed.")
        return jsonify({
            "error": "The chatbot could not process your request right now."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
