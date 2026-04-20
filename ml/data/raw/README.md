# ML Training Data

## Phase 3: Generating Synthetic Training Data

The `generate_synthetic_data.py` script uses Gemini to generate training samples for both models.

### Prerequisites

```bash
pip install google-generativeai
```

### Step 1: Generate feedback turn classification data

```bash
python ml/data/raw/generate_synthetic_data.py \
  --api-key YOUR_GEMINI_API_KEY \
  --output ml/data/raw/feedback_turn_classification.jsonl \
  --count 1500 \
  --type feedback
```

### Step 2: Generate conversation policy data

```bash
python ml/data/raw/generate_synthetic_data.py \
  --api-key YOUR_GEMINI_API_KEY \
  --output ml/data/raw/conversation_policy.jsonl \
  --count 800 \
  --type policy
```

### Step 3: Build train/val/test splits

```bash
python ml/training/build_splits.py \
  --source ml/data/raw/feedback_turn_classification.jsonl \
  --output-dir ml/data/processed/feedback_turns/

python ml/training/build_splits.py \
  --source ml/data/raw/conversation_policy.jsonl \
  --output-dir ml/data/processed/policy/
```

### Step 4: Train both models

```bash
cd ml
python training/train_all.py
```

### Step 5: Start the inference server

```bash
cd ml
uvicorn serve.inference_server:app --host 0.0.0.0 --port 8001 --reload
```

### Step 6: Set backend env to use ML mode

In your backend `.env`:
```
LLM_PROVIDER_MODE=ml
ML_MODEL_URL=http://localhost:8001
GEMINI_API_KEY=your-key-here  # Still needed for question generation
```

## Architecture

In `ml` mode, the system works as:
1. **DistilRoBERTa feedback model** → classifies user sentiment, themes, quality, continue signal
2. **DistilRoBERTa policy model** → selects the next interview action (probe_weakness, etc.)
3. **Gemini** → generates the actual contextual question text based on the policy + full context

This hybrid gives you the best of both worlds: local, fast, private classification + Gemini's natural language quality.
