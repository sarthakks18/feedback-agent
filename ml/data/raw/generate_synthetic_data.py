"""
generate_synthetic_data.py — Phase 3: ML Model Training Data Generator

Uses Gemini to generate high-quality JSONL training samples.

Two modes (--type flag):
  feedback  — Generates feedback turn classification data (for train_multitask.py)
  policy    — Generates conversation policy data (for train_policy_model.py)

Usage:
  # Feedback classification data (1500 samples)
  python ml/data/raw/generate_synthetic_data.py \\
    --api-key YOUR_GEMINI_API_KEY \\
    --output ml/data/raw/feedback_turn_classification.jsonl \\
    --count 1500 --type feedback

  # Policy decision data (800 samples)
  python ml/data/raw/generate_synthetic_data.py \\
    --api-key YOUR_GEMINI_API_KEY \\
    --output ml/data/raw/conversation_policy.jsonl \\
    --count 800 --type policy

Then run: python ml/training/build_splits.py and python training/train_all.py
"""

import argparse
import json
import random
import time
from pathlib import Path

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Label space (mirrors label_maps.py exactly)
# ---------------------------------------------------------------------------

SENTIMENT_LABELS = ["positive", "neutral", "hesitant", "frustrated", "wants_to_stop"]
TOPIC_LABELS = ["on_topic", "off_topic"]
CONTINUE_LABELS = ["continue", "uncertain", "stop"]
QUALITY_LABELS = ["vague", "somewhat_actionable", "highly_actionable"]
SESSION_STAGE_LABELS = ["opening", "first_impression", "strengths", "weaknesses", "improvement_request", "wrap_up"]
THEME_LABELS = [
    "accuracy", "completeness", "relevance", "clarity", "tone",
    "formatting", "hallucination", "latency", "usability", "reasoning",
    "instruction_following", "safety", "other",
]
POLICY_LABELS = [
    "greet_opening", "ask_first_impression", "probe_strength", "probe_weakness",
    "probe_specific_issue", "ask_improvement_priority", "redirect_to_feedback",
    "shorten_question", "empathy_then_probe", "confirm_end", "wrap_up",
]

SOURCE_MODELS = ["GPT-4", "Claude 3", "Gemini Pro", "Llama 3", "Mistral", "DALL-E 3", "Stable Diffusion", "Whisper"]

# ---------------------------------------------------------------------------
# Scenarios for variety across content types
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "input_type": "code",
        "original_prompt": "Write a Python function to merge two sorted linked lists.",
        "generated_content": "def merge(l1, l2):\n    result = []\n    while l1 and l2:\n        if l1.val < l2.val:\n            result.append(l1.val)\n            l1 = l1.next\n        else:\n            result.append(l2.val)\n            l2 = l2.next\n    return result  # BUG: doesn't reconstruct linked list",
    },
    {
        "input_type": "text",
        "original_prompt": "Explain quantum entanglement to a 10-year-old.",
        "generated_content": "Quantum entanglement is when two particles become linked so that measuring one instantly affects the other, no matter how far apart they are. It's like magic dice that always show opposite faces.",
    },
    {
        "input_type": "image",
        "original_prompt": "Generate a professional logo for a fintech startup called 'Nexus Pay'.",
        "generated_content": "[Image: A blue circular logo with 'NP' monogram and lightning bolt icon]",
    },
    {
        "input_type": "audio",
        "original_prompt": "Create a 30-second podcast intro for a technology show.",
        "generated_content": "[Audio: Upbeat electronic music with voiceover: Welcome to TechForward, where tomorrow's innovations come alive today...]",
    },
    {
        "input_type": "pdf",
        "original_prompt": "Write a business plan for a sustainable clothing brand.",
        "generated_content": "[Document: 8-page business plan with executive summary, market analysis, and financial projections for EcoThread brand]",
    },
    {
        "input_type": "code",
        "original_prompt": "Create a REST API endpoint for user authentication using JWT.",
        "generated_content": "app.post('/login', (req, res) => {\n  const { user } = req.body;\n  const token = jwt.sign({ user }, 'secret');\n  res.json({ token });\n});  // NOTE: hardcoded secret, no password verification",
    },
    {
        "input_type": "text",
        "original_prompt": "Summarize the key findings of a climate change report.",
        "generated_content": "Global temperatures have risen 1.1 degrees since pre-industrial times. Arctic ice is melting at unprecedented rates. Extreme weather events are 40% more frequent.",
    },
    {
        "input_type": "video",
        "original_prompt": "Create a 60-second product demo video for a smart home device.",
        "generated_content": "[Video: Animated product showcase showing smart speaker controlling home devices, narrated with professional voice-over]",
    },
]


# ---------------------------------------------------------------------------
# Label sampling with logical consistency
# ---------------------------------------------------------------------------

def sample_labels(rng: random.Random) -> dict:
    sentiment = rng.choice(SENTIMENT_LABELS)
    continue_signal = "stop" if sentiment == "wants_to_stop" else (
        "uncertain" if sentiment in ["hesitant", "frustrated"] and rng.random() > 0.5 else "continue"
    )
    topic = "off_topic" if rng.random() < 0.12 else "on_topic"
    quality = "vague" if sentiment in ["hesitant", "wants_to_stop"] else (
        "highly_actionable" if sentiment == "positive" and rng.random() > 0.5 else "somewhat_actionable"
    )
    stage = rng.choice(SESSION_STAGE_LABELS)
    num_themes = rng.randint(1, 3)
    themes = rng.sample(THEME_LABELS[:-1], min(num_themes, len(THEME_LABELS) - 1))

    return {
        "sentiment": sentiment,
        "topic": topic,
        "continue_signal": continue_signal,
        "feedback_quality": quality,
        "session_stage": stage,
        "themes": themes,
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_feedback_prompt(scenario: dict, labels: dict) -> str:
    return f"""Generate a realistic user message for a feedback interview session.

The user is giving feedback on this AI-generated content:
- Input Type: {scenario['input_type']}
- Original Prompt: "{scenario['original_prompt']}"
- Generated Content: "{scenario['generated_content'][:300]}"

The user message MUST reflect:
- Sentiment: {labels['sentiment']}
- Topic: {labels['topic']}
- Continue Signal: {labels['continue_signal']}
- Feedback Quality: {labels['feedback_quality']}
- Session Stage: {labels['session_stage']}
- Themes: {', '.join(labels['themes'])}

Write a natural, realistic message (1-4 sentences) like a real person giving feedback.

Return ONLY valid JSON:
{{
  "latest_user_message": "<the user message>",
  "recent_context": [
    {{"role": "assistant", "text": "<a realistic preceding interviewer question>"}},
    {{"role": "user", "text": "<a preceding user answer>"}}
  ]
}}"""


def build_policy_prompt(scenario: dict, signals: dict, target_policy: str) -> str:
    return f"""Generate a realistic conversation context for a policy model training sample.

Interview situation:
- Session stage: {signals['session_stage']}
- User sentiment: {signals['sentiment']}
- Topic relevance: {signals['topic']}
- Continue signal: {signals['continue_signal']}
- Feedback quality: {signals['feedback_quality']}
- Themes mentioned: {', '.join(signals['themes'])}

The correct next interviewer action is: {target_policy}

Generate a realistic recent conversation (2-4 turns) that leads to this situation.

Return ONLY valid JSON:
{{
  "recent_context": [
    {{"role": "assistant", "text": "<interviewer message>"}},
    {{"role": "user", "text": "<user response>"}}
  ]
}}"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic FeedbackAI training data using Gemini.")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--count", type=int, default=1500, help="Number of samples to generate")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model to use")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between API calls (seconds)")
    parser.add_argument("--type", choices=["feedback", "policy"], default="feedback",
                        dest="data_type", help="Dataset type to generate")
    args = parser.parse_args()

    genai.configure(api_key=args.api_key)
    gemini_model = genai.GenerativeModel(
        args.model,
        generation_config=genai.types.GenerationConfig(
            temperature=0.9,
            response_mime_type="application/json",
        ),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    generated = 0
    failed = 0

    print(f"Generating {args.count} '{args.data_type}' samples → {output_path}")

    with open(output_path, "w", encoding="utf-8") as fout:
        sample_id = 0
        while generated < args.count:
            scenario = rng.choice(SCENARIOS)
            source_model = rng.choice(SOURCE_MODELS)
            labels = sample_labels(rng)

            try:
                if args.data_type == "feedback":
                    prompt = build_feedback_prompt(scenario, labels)
                    response = gemini_model.generate_content(prompt)
                    raw = response.text.strip()
                    parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])

                    record = {
                        "sample_id": f"syn_{sample_id:05d}",
                        "input_type": scenario["input_type"],
                        "source_model_label": source_model,
                        "session_stage": labels["session_stage"],
                        "original_prompt": scenario["original_prompt"],
                        "generated_content": scenario["generated_content"][:600],
                        "recent_context": parsed.get("recent_context", []),
                        "latest_user_message": parsed["latest_user_message"],
                        "labels": {
                            "sentiment": labels["sentiment"],
                            "topic": labels["topic"],
                            "continue_signal": labels["continue_signal"],
                            "feedback_quality": labels["feedback_quality"],
                            "session_stage": labels["session_stage"],
                            "themes": labels["themes"],
                        },
                    }

                else:  # policy
                    target_policy = rng.choice(POLICY_LABELS)
                    signals = {
                        "sentiment": labels["sentiment"],
                        "topic": labels["topic"],
                        "continue_signal": labels["continue_signal"],
                        "feedback_quality": labels["feedback_quality"],
                        "session_stage": labels["session_stage"],
                        "themes": labels["themes"],
                    }
                    prompt = build_policy_prompt(scenario, signals, target_policy)
                    response = gemini_model.generate_content(prompt)
                    raw = response.text.strip()
                    parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])

                    record = {
                        "sample_id": f"pol_{sample_id:05d}",
                        "session_stage": labels["session_stage"],
                        "recent_context": parsed.get("recent_context", []),
                        "model_signals": signals,
                        "target_policy": target_policy,
                    }

                fout.write(json.dumps(record, ensure_ascii=True) + "\n")
                generated += 1
                sample_id += 1

                if generated % 50 == 0:
                    print(f"  [{generated}/{args.count}] {generated} samples written ({failed} failed)")

                time.sleep(args.delay)

            except Exception as exc:
                failed += 1
                print(f"  [WARN] Sample {sample_id} failed: {exc}")
                if failed > args.count * 0.2:
                    print("[ERROR] Too many failures, stopping.")
                    break
                time.sleep(1.0)

    print(f"\nDone. {generated} samples written ({failed} failures).")
    print(f"Next: python ml/training/build_splits.py --source {output_path} --output-dir ml/data/processed/")


if __name__ == "__main__":
    main()
