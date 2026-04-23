import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { env } from "../config/env.js";
import { HttpError } from "../lib/http-error.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------------------------------------------------------
// Config helpers
// ---------------------------------------------------------------------------

function getApiKey() {
  const key = env.geminiApiKey || env.llmApiKey;
  if (!key) throw new HttpError(500, "No Gemini API key configured. Set GEMINI_API_KEY in your environment.");
  return key;
}

// Model preference order: env override → flash-lite (higher free-tier quota) → flash
const MODEL_PREFERENCE = [
  env.geminiModel,
  env.llmModel,
  "gemini-2.0-flash-lite",
  "gemini-2.0-flash",
].filter(Boolean);

function getModel(index = 0) {
  return MODEL_PREFERENCE[Math.min(index, MODEL_PREFERENCE.length - 1)];
}

// ---------------------------------------------------------------------------
// Core Gemini caller — uses fetch against OpenAI-compat endpoint so zero
// extra dependencies are needed (works with the existing LLM_API_URL/KEY).
// Falls back to direct Gemini REST if LLM_API_URL is not set.
// ---------------------------------------------------------------------------

async function callGeminiRaw(prompt, { temperature = 0.5, isJson = true, modelIndex = 0, filePart = null } = {}) {
  const apiKey = getApiKey();
  const model = getModel(modelIndex);
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

  const parts = [];
  if (filePart) parts.push(filePart);
  parts.push({ text: prompt });

  const body = {
    contents: [{ role: "user", parts }],
    generationConfig: {
      temperature,
      candidateCount: 1,
      maxOutputTokens: 256,
      ...(isJson ? { responseMimeType: "application/json" } : {}),
    },
  };

  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    // On 429 try next model in preference list once before giving up
    if (resp.status === 429 && modelIndex < MODEL_PREFERENCE.length - 1) {
      console.warn(`[llm] 429 on ${model}, retrying with ${getModel(modelIndex + 1)}`);
      await new Promise(r => setTimeout(r, 1500));
      return callGeminiRaw(prompt, { temperature, isJson, modelIndex: modelIndex + 1, filePart });
    }
    throw new Error(`Gemini API error ${resp.status}: ${errText}`);
  }

  const data = await resp.json();
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text ?? "";
  return text;
}

async function callGeminiJson(prompt, { temperature = 0.5, filePart = null } = {}) {
  const text = await callGeminiRaw(prompt, { temperature, isJson: true, filePart });
  try {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start === -1 || end === -1) throw new Error("No JSON found");
    return JSON.parse(text.slice(start, end + 1));
  } catch {
    throw new HttpError(502, "Gemini returned non-JSON output", text);
  }
}

// ---------------------------------------------------------------------------
// Multimodal file helper (Phase 2) — reads uploaded file and returns a
// Gemini REST inlineData part
// ---------------------------------------------------------------------------

const VERCEL_UPLOADS = "/tmp/uploads";
const LOCAL_UPLOADS = path.resolve(__dirname, "../../../uploads");

async function buildFilePart(submission) {
  if (!submission?.uploadedFileUrl || !submission?.uploadedFileType) return null;

  const filename = submission.uploadedFileUrl.replace(/^\/uploads\//, "");
  const isVercel = process.env.VERCEL === "1" || !!process.env.VERCEL;
  const uploadsRoot = isVercel ? VERCEL_UPLOADS : LOCAL_UPLOADS;
  const filePath = path.join(uploadsRoot, filename);

  if (!fs.existsSync(filePath)) {
    console.warn(`[llm] File not found at ${filePath}, skipping multimodal injection`);
    return null;
  }

  try {
    const data = fs.readFileSync(filePath).toString("base64");
    const mimeType = normalizeMime(submission.uploadedFileType);
    return { inlineData: { mimeType, data } };
  } catch (err) {
    console.warn(`[llm] Could not read file ${filePath}:`, err.message);
    return null;
  }
}

function normalizeMime(rawMime) {
  const map = {
    "image/jpg": "image/jpeg",
    "audio/mpeg": "audio/mp3",
    "audio/x-m4a": "audio/mp4",
    "video/quicktime": "video/mp4",
  };
  return map[rawMime] ?? rawMime ?? "application/octet-stream";
}

// ---------------------------------------------------------------------------
// Build the full context-aware prompt that Gemini sees on each turn
// ---------------------------------------------------------------------------

function buildInterviewerPrompt({ submission, conversationHistory, latestUserMessage, coveredThemes, policy, questionNumber }) {
  // Keep last 8 turns max to stay within free-tier token limits but maintain context
  const recentHistory = conversationHistory.slice(-8);
  const historyBlock = recentHistory.length > 0
    ? recentHistory.map(m => `${m.role === "assistant" ? "AI Assistant" : "User"}: ${m.content}`).join("\n")
    : "";

  const contentSnippet = submission.uploadedFileUrl
    ? `[Uploaded ${submission.inputType} file]`
    : (submission.generatedContent ?? "").slice(0, 300);

  const notCovered = coveredThemes.length > 0 ? `Skip: ${coveredThemes.join(", ")}.` : "";
  const policyHint = policy ? `Focus: ${policyToInstruction(policy)}` : "";

  return `# Revised Strategic Model: Intelligent Visual Generation Feedback System

You are an intelligent, empathetic AI feedback assistant assessing an image/visual generation model.

1. Interactive Visual Dialogue Architecture
Instead of fixed questionnaires, act as a responsive visual discussion engine. Interpret vague dissatisfaction (e.g., "It looks fake"). Break down abstract complaints into measurable components (placement, texture, quantity, lighting). Track mentioned issues and ask refinement-based questions to convert subjective dissatisfaction into technical feedback.

2. Visual Consistency Evaluation Layer
Analyze image quality across:
A. Subject–Environment Relationship: Does the object naturally belong?
B. Lighting & Illumination Balance: Consistent sources across characters, props, background, shadows.
C. Cohesion of Visual Elements: Same spatial and atmospheric conditions?
D. Realism vs. Stylization Alignment: Is the fantasy visually believable?

3. User-Intent Anchored Evaluation
Ask what the intended outcome was (realistic, cinematic, abstract) and goal (functional/aesthetic). Different objectives change evaluation weights.

4. Adaptive Emotional Calibration
Detect tone and respond:
- Mild disappointment -> Clarifying/solution-oriented
- Frustration -> Empathetic, acknowledging mismatch
- Positive surprise -> Reinforcing, probing for replicable strengths

5. Structured Visual Rating
Maintain a 1-5 scale for comparability (1: Very poor, 2: Below expectations, 3: Acceptable but needs improvement, 4: Good with minor issues, 5: Excellent).

6. Prompt & Rendering Context Capture
Isolate root-causes (was an issue due to prompt ambiguity or rendering limits?).

7. Failure Localization Mechanism
Isolate exactly: Which object? Which region? Which attribute?

8. Visual Credibility & Trust Audit
Evaluate expectation alignment (spatial logic, physics).

9. Dynamic Question Generation
- DO NOT follow any fixed list of questions.
- Your NEXT question must ALWAYS be directly based on the user's PREVIOUS answer. Dig deeper into whatever they just said.
- If the user says "The lighting is weird", ask exactly what part of the lighting is weird or how they expected it to look.
- Start your response with a quick, empathetic acknowledgment of their answer.
- Ask ONLY ONE focused question at a time.
- Keep responses natural, conversational, and human (under 3 sentences).

10. Dynamic Conversation End
- Keep asking dynamic, follow-up questions until you feel you have fully understood their feedback, root cause, and suggestions.
- THERE IS NO FIXED QUESTION LIMIT. Keep going as long as the user provides meaningful feedback.
- ONCE you have collected complete, actionable feedback, give a brief final wrap-up thanking the user and set "shouldEnd": true.
- If the user explicitly asks to stop or end the chat, immediately thank them and set "shouldEnd": true.

─────────────────────────────────────────────────
CURRENT SESSION CONTEXT
─────────────────────────────────────────────────
User's Target Generation Goal: ${submission.title}
Input Type: ${submission.inputType}
Original Prompt: "${(submission.originalPrompt ?? "").slice(0, 300)}"
AI Output Context: "${contentSnippet}"

Conversation so far:
${historyBlock}
User just said: "${latestUserMessage}"

${notCovered} ${policyHint}

Your job: Write ONE short follow-up response (max 2-3 sentences) as the AI feedback assistant.
1. Directly acknowledge with a quick appreciation or apology for the user's last answer (e.g. "Thanks for pointing that out!", "I'm sorry to hear that.").
2. Ask the next logical diagnostic question based on the framework above (unless you are ending the session).
3. If you have gathered all the perfect feedback, summarize slightly, thank them, and set shouldEnd to true.
4. Never ask two questions at once.
5. Only output JSON.

Return ONLY this JSON (no markdown):
{"reply": "<appreciation/apology + next question OR closing wrap-up>", "shouldEnd": false, "topicCovered": "<rating|goal|smoothness|strengths|main_issue|localization|context|severity|reuse|suggestions|other>"}`;
}

function policyToInstruction(policy) {
  const map = {
    greet_opening: "Ask for the user's very first impression.",
    ask_first_impression: "Ask how well the output matched their original prompt.",
    probe_strength: "Ask what felt strongest or most useful.",
    probe_weakness: "Ask where the output fell short — probe a specific gap.",
    probe_specific_issue: "Ask them to describe the issue they mentioned in concrete detail.",
    ask_improvement_priority: "Ask what single improvement would make this most valuable.",
    redirect_to_feedback: "Politely bring them back to evaluating the AI output.",
    shorten_question: "Ask one brief, easy question.",
    empathy_then_probe: "Acknowledge their frustration warmly, then ask one follow-up.",
    confirm_end: "Confirm whether they want to end the session.",
    wrap_up: "Thank them and close the session.",
  };
  return map[policy] ?? "Ask a thoughtful follow-up about the user's experience with the output.";
}

// ---------------------------------------------------------------------------
// ML inference server helpers (Phase 3 strategy layer)
// ---------------------------------------------------------------------------

async function callMlServer(endpoint, body) {
  if (!env.mlModelUrl) throw new Error("ML_MODEL_URL not configured");
  const resp = await fetch(`${env.mlModelUrl}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`ML server ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

// ---------------------------------------------------------------------------
// Local fallback analysis (no API needed)
// ---------------------------------------------------------------------------

function analyzeIntentLocally(message) {
  const m = message.toLowerCase();
  if (/\b(stop|end|done|enough|quit|finish|bye|wrap up)\b/.test(m))
    return { sentimentLabel: "wants_to_stop", continueSignal: "stop", offTopic: false, themes: [] };
  if (/\b(frustrated|annoyed|hate|terrible|awful|waste|useless)\b/.test(m))
    return { sentimentLabel: "frustrated", continueSignal: "uncertain", offTopic: false, themes: [] };
  if (/\b(not sure|maybe|unsure|i guess|tired|busy|short)\b/.test(m))
    return { sentimentLabel: "hesitant", continueSignal: "uncertain", offTopic: false, themes: [] };
  if (/\b(weather|sports|movie|recipe|joke|news|game)\b/.test(m))
    return { sentimentLabel: "neutral", continueSignal: "continue", offTopic: true, themes: [] };
  return { sentimentLabel: "engaged", continueSignal: "continue", offTopic: false, themes: [] };
}

function buildFallbackQuestion(submission, analysis, questionNumber, coveredThemes) {
  if (analysis.continueSignal === "stop") {
    return { reply: `Thank you so much for your feedback on "${submission.title}". I'll prepare a summary now.`, shouldEnd: true, topicCovered: "other" };
  }
  if (analysis.offTopic) {
    return { reply: `Let's keep focused on the AI output. What did you think about how well it addressed your original request?`, shouldEnd: false, topicCovered: "relevance" };
  }
  const pool = [
    { q: "What was your initial reaction when you first saw the generated output?", t: "first_impression" },
    { q: "What part of the output worked best for your needs?", t: "strengths" },
    { q: "Where exactly did the output fall short of what you expected?", t: "weaknesses" },
    { q: "If you could change one thing about this output, what would have the biggest impact?", t: "improvement_request" },
    { q: "How well did the output match the tone and style you were looking for?", t: "tone" },
    { q: "Were there any factual errors or inaccuracies in the output?", t: "accuracy" },
  ];
  const available = pool.filter(p => !coveredThemes.includes(p.t));
  const chosen = available[(questionNumber - 1) % Math.max(available.length, 1)] ?? pool[pool.length - 1];
  return { reply: chosen.q, shouldEnd: false, topicCovered: chosen.t };
}

function buildLocalSummary(submission, messages) {
  const userMessages = messages.filter(m => m.role === "user");
  const combined = userMessages.map(m => m.content).join(" ").toLowerCase();

  const hasPositive = /\b(good|great|clear|helpful|accurate|loved|perfect|excellent)\b/.test(combined);
  const hasNegative = /\b(bad|wrong|unclear|missed|slow|incorrect|halluc|confus)\b/.test(combined);
  const hasImprovement = /\b(should|could|improve|better|change|fix|add|missing)\b/.test(combined);

  return {
    shortSummary: `Feedback session for "${submission.title}" — ${userMessages.length} response(s) collected on the ${submission.inputType} output from ${submission.sourceModelLabel}.`,
    strengths: hasPositive ? ["User highlighted positive aspects of the generated output."] : ["No strong positive feedback captured."],
    weaknesses: hasNegative ? ["User identified quality gaps or mismatches in the output."] : ["No critical weaknesses were explicitly stated."],
    recommendations: hasImprovement ? ["Address the specific improvement mentioned by the user."] : ["Gather more targeted feedback in future sessions."],
    sentimentTimeline: userMessages.map((m, i) => ({ step: i + 1, sentiment: m.sentimentLabel || "neutral" })),
    engagementLevel: userMessages.length >= 3 ? "high" : userMessages.length === 2 ? "medium" : "low",
    summaryConfidence: userMessages.length >= 2 ? "medium" : "low",
    overallSentiment: userMessages.at(-1)?.sentimentLabel ?? "neutral",
    continueSignalFinal: "stop",
  };
}

// ---------------------------------------------------------------------------
// EXPORTED: analyzeUserTurn
// ---------------------------------------------------------------------------

export async function analyzeUserTurn({ message, submission, recentContext, sessionStage }) {
  // ML mode — local model classifies
  if (env.llmProviderMode === "ml") {
    try {
      const result = await callMlServer("/analyze-turn", {
        input_type: submission?.inputType ?? "text",
        source_model_label: submission?.sourceModelLabel ?? "unknown",
        session_stage: sessionStage ?? "weaknesses",
        original_prompt: submission?.originalPrompt ?? "",
        generated_content: submission?.generatedContent ?? "",
        recent_context: (recentContext ?? []).map(m => ({ role: m.role, text: m.content })),
        latest_user_message: message,
      });
      return {
        sentimentLabel: result.sentiment,
        continueSignal: result.continue_signal,
        offTopic: result.topic === "off_topic",
        themes: result.themes || [],
        feedbackQuality: result.feedback_quality,
        sessionStage: result.session_stage,
      };
    } catch (err) {
      console.warn("[llm] ML server unreachable, falling back to local:", err.message);
    }
  }

  // API mode — ask Gemini to classify
  if (env.llmProviderMode === "api") {
    try {
      const classifyPrompt = `Analyze this user message from a feedback interview about an AI-generated ${submission?.inputType ?? "output"}.

User said: "${message}"
Their original prompt to AI was: "${(submission?.originalPrompt ?? "").slice(0, 300)}"
Session stage: ${sessionStage ?? "unknown"}

Return ONLY valid JSON:
{
  "sentimentLabel": "<positive|neutral|hesitant|frustrated|wants_to_stop>",
  "continueSignal": "<continue|uncertain|stop>",
  "offTopic": <true|false>,
  "themes": ["<accuracy|completeness|relevance|clarity|tone|formatting|hallucination|usability|reasoning|improvement_request|other>"],
  "feedbackQuality": "<vague|somewhat_actionable|highly_actionable>"
}`;

      const result = await callGeminiJson(classifyPrompt, { temperature: 0.1 });
      return result;
    } catch (err) {
      console.warn("[llm] Gemini analysis failed, falling back to local:", err.message);
    }
  }

  return analyzeIntentLocally(message);
}

// ---------------------------------------------------------------------------
// EXPORTED: generateInterviewerReply
// ---------------------------------------------------------------------------

export async function generateInterviewerReply({
  submission,
  userMessage,
  analysis,
  questionNumber,
  recentContext,
  sessionStage,
  coveredThemes = [],
}) {
  // Hard stop — always honour
  if (analysis?.continueSignal === "stop") {
    return {
      reply: `Thank you so much for taking the time to share your feedback on "${submission.title}". I'll now prepare a summary of our conversation.`,
      shouldEnd: true,
      topicCovered: "other",
    };
  }

  // Get policy from ML server if in ml mode
  let policy = null;
  if (env.llmProviderMode === "ml") {
    try {
      const signals = {
        sentiment: analysis?.sentimentLabel ?? "neutral",
        topic: analysis?.offTopic ? "off_topic" : "on_topic",
        continue_signal: analysis?.continueSignal ?? "continue",
        themes: analysis?.themes ?? [],
        feedback_quality: analysis?.feedbackQuality ?? "somewhat_actionable",
      };
      const result = await callMlServer("/select-policy", {
        session_stage: sessionStage ?? "weaknesses",
        recent_context: (recentContext ?? []).map(m => ({ role: m.role, text: m.content })),
        model_signals: signals,
      });
      policy = result.policy ?? null;
    } catch (err) {
      console.warn("[llm] ML policy server unreachable:", err.message);
    }
  }

  // API or ML mode — use Gemini for question generation
  if (env.llmProviderMode === "api" || env.llmProviderMode === "ml") {
    try {
      // Build conversation history (all prior turns excluding the latest user message)
      const priorMessages = (recentContext ?? []).slice(0, -1);

      const prompt = buildInterviewerPrompt({
        submission,
        conversationHistory: priorMessages,
        latestUserMessage: userMessage,
        coveredThemes,
        policy,
        questionNumber,
      });

      // For multimodal: if file exists and it's an early turn, append note
      // (file is referenced in the prompt text; actual base64 injection is
      //  skipped on Vercel because files don't persist across serverless calls)
      const filePart = await buildFilePart(submission);
      const result = await callGeminiJson(prompt, { temperature: 0.6, filePart });

      return {
        reply: result.reply || "Thank you for sharing that. Could you tell me more?",
        shouldEnd: Boolean(result.shouldEnd),
        topicCovered: result.topicCovered || "other",
      };
    } catch (err) {
      console.warn("[llm] Gemini reply generation failed, using fallback:", err.message);
    }
  }

  // Mock / fallback
  return buildFallbackQuestion(submission, analysis, questionNumber, coveredThemes);
}

// ---------------------------------------------------------------------------
// EXPORTED: generateOpeningMessage
// ---------------------------------------------------------------------------

export async function generateOpeningMessage(submission) {
  if (env.llmProviderMode === "api" || env.llmProviderMode === "ml") {
    try {
      const filePart = await buildFilePart(submission);
      const inputLabel =
        submission.inputType === "image" ? "image"
        : submission.inputType === "audio" ? "audio recording"
        : submission.inputType === "video" ? "video"
        : submission.inputType === "pdf" ? "document"
        : submission.inputType === "code" ? "code output"
        : "response";
        
      const prompt = `You are an AI feedback assistant. A user has submitted an AI-generated ${inputLabel} for feedback.
Target Goal / Prompt: "${submission.title}"
${submission.originalPrompt ? `Detailed Prompt: "${submission.originalPrompt.slice(0, 300)}"` : ""}

Your task is to start the feedback session.
1. If an image or file is attached/uploaded, analyze it deeply.
2. Generate an opening message that references specific details you see in the image/output. 
3. Ask the user ONE focused question about how well it met their target goal, based on what you actually observe in the output.
4. Keep it friendly and concise (max 2-3 sentences).
5. Only output JSON.

Return ONLY this JSON:
{"reply": "<your contextual opening message>", "topicCovered": "first_impression"}
`;

      const result = await callGeminiJson(prompt, { temperature: 0.6, filePart });
      if (result && result.reply) {
        return { reply: result.reply, topicCovered: result.topicCovered || "first_impression" };
      }
    } catch (err) {
      console.warn("[llm] Gemini opening message failed, using fallback:", err.message);
    }
  }

  // Fallback
  const inputLabel =
    submission.inputType === "image" ? "image"
    : submission.inputType === "audio" ? "audio recording"
    : submission.inputType === "video" ? "video"
    : submission.inputType === "pdf" ? "document"
    : submission.inputType === "code" ? "code output"
    : "response";
    
  return {
    reply: `Hello, and welcome to this feedback session about "${submission.title}". I'm here to collect your thoughts on the AI-generated ${inputLabel} you submitted. To start — what was your very first impression when you saw the output?`,
    topicCovered: "first_impression"
  };
}

// ---------------------------------------------------------------------------
// EXPORTED: generateSessionSummary
// ---------------------------------------------------------------------------

export async function generateSessionSummary({ submission, messages }) {
  if (env.llmProviderMode === "api" || env.llmProviderMode === "ml") {
    try {
      const transcript = messages
        .map(m => `${m.role === "assistant" ? "INTERVIEWER" : "USER"}: ${m.content}`)
        .join("\n");

      const summaryPrompt = `You are a feedback analyst. Summarise this feedback interview.

Submission: "${submission.title}" (${submission.inputType} from ${submission.sourceModelLabel})
Original prompt: "${(submission.originalPrompt ?? "").slice(0, 400)}"

Interview transcript:
${transcript.slice(0, 3500)}

Return ONLY valid JSON:
{
  "shortSummary": "<2-3 sentence summary>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "recommendations": ["<recommendation 1>", "<recommendation 2>"],
  "sentimentTimeline": [{"step": 1, "sentiment": "<label>"}],
  "engagementLevel": "<low|medium|high>",
  "summaryConfidence": "<low|medium|high>",
  "overallSentiment": "<positive|neutral|hesitant|frustrated|wants_to_stop>",
  "continueSignalFinal": "stop"
}`;

      const result = await callGeminiJson(summaryPrompt, { temperature: 0.3 });
      return result;
    } catch (err) {
      console.warn("[llm] Gemini summary failed, using local builder:", err.message);
    }
  }

  return buildLocalSummary(submission, messages);
}
