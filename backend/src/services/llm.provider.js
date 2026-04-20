import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenerativeAI } from "@google/generative-ai";

import { env } from "../config/env.js";
import { HttpError } from "../lib/http-error.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---------------------------------------------------------------------------
// Gemini SDK client (initialised lazily so missing key doesn't crash on startup)
// ---------------------------------------------------------------------------

let _geminiClient = null;

function getGeminiClient() {
  if (!_geminiClient) {
    const key = env.geminiApiKey || env.llmApiKey;
    if (!key) {
      throw new HttpError(500, "No Gemini API key configured. Set GEMINI_API_KEY in your environment.");
    }
    _geminiClient = new GoogleGenerativeAI(key);
  }
  return _geminiClient;
}

// ---------------------------------------------------------------------------
// ML inference server helpers (Phase 3 — strategy layer)
// ---------------------------------------------------------------------------

async function callMlServer(endpoint, body) {
  if (!env.mlModelUrl) {
    throw new Error("ML_MODEL_URL is not configured");
  }
  const response = await fetch(`${env.mlModelUrl}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`ML server error ${response.status}: ${text}`);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// File → Gemini inline_data helper (Phase 2 — multimodal injection)
// ---------------------------------------------------------------------------

const VERCEL_UPLOADS = "/tmp/uploads";
const LOCAL_UPLOADS = path.resolve(__dirname, "../../../uploads");

/**
 * Converts a stored uploadedFileUrl like "/uploads/1234-foo.png"
 * into a Gemini-compatible { inlineData: { mimeType, data } } part.
 * Returns null if the file cannot be resolved.
 */
async function buildFilePart(submission) {
  if (!submission.uploadedFileUrl || !submission.uploadedFileType) {
    return null;
  }

  // Strip leading /uploads/ to get just the filename
  const filename = submission.uploadedFileUrl.replace(/^\/uploads\//, "");
  const isVercel = process.env.VERCEL === "1" || !!process.env.VERCEL;
  const uploadsRoot = isVercel ? VERCEL_UPLOADS : LOCAL_UPLOADS;
  const filePath = path.join(uploadsRoot, filename);

  if (!fs.existsSync(filePath)) {
    console.warn(`[llm.provider] File not found at ${filePath}, skipping inline injection`);
    return null;
  }

  try {
    const data = fs.readFileSync(filePath).toString("base64");
    const mimeType = normalizeMime(submission.uploadedFileType);
    return { inlineData: { mimeType, data } };
  } catch (err) {
    console.warn(`[llm.provider] Failed to read file ${filePath}:`, err.message);
    return null;
  }
}

function normalizeMime(rawMime) {
  if (!rawMime) return "application/octet-stream";
  // Map common variants to Gemini-supported MIME types
  const map = {
    "image/jpg": "image/jpeg",
    "audio/mpeg": "audio/mp3",
    "audio/x-m4a": "audio/mp4",
    "video/quicktime": "video/mp4",
    "application/octet-stream": "application/octet-stream",
  };
  return map[rawMime] ?? rawMime;
}

// ---------------------------------------------------------------------------
// Conversation history builder
// ---------------------------------------------------------------------------

/**
 * Converts stored SessionMessage rows into Gemini `contents[]` format.
 * The system prompt is handled separately via systemInstruction, so we
 * only include user↔model turns here.
 */
function buildGeminiHistory(messages) {
  return messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    }));
}

// ---------------------------------------------------------------------------
// Context-aware system prompt builder (Phase 1 core)
// ---------------------------------------------------------------------------

/**
 * Builds the rich structured system prompt that grounds Gemini in the
 * actual submission content and conversation state.
 */
function buildContextualSystemPrompt(submission, coveredThemes = [], policy = null) {
  const contentLabel =
    submission.inputType === "image"
      ? "AI-GENERATED IMAGE"
      : submission.inputType === "audio"
      ? "AI-GENERATED AUDIO"
      : submission.inputType === "video"
      ? "AI-GENERATED VIDEO"
      : submission.inputType === "pdf"
      ? "AI-GENERATED DOCUMENT"
      : submission.inputType === "code"
      ? "AI-GENERATED CODE"
      : "AI-GENERATED RESPONSE";

  const contentSection =
    submission.uploadedFileUrl
      ? `[${contentLabel}]\nA file has been provided inline above. Examine its content carefully when forming questions.`
      : `[${contentLabel}]\n${submission.generatedContent?.slice(0, 1200) || "No content available."}`;

  const policyInstruction = policy
    ? `[CURRENT INTERVIEW FOCUS]\n${policyToInstruction(policy)}`
    : `[CURRENT INTERVIEW FOCUS]\nProbe the user's overall impression — ask an open, curious question.`;

  const themeBlock =
    coveredThemes.length > 0
      ? `[TOPICS ALREADY EXPLORED]\n${coveredThemes.join(", ")}\nDo NOT repeat questions on these topics. Explore something new.`
      : `[TOPICS ALREADY EXPLORED]\nNone yet. Start with the user's first impression.`;

  return `You are a professional, empathetic feedback interviewer. Your sole purpose is to collect structured, actionable feedback on AI-generated content.

[ORIGINAL USER PROMPT GIVEN TO THE AI]
${submission.originalPrompt?.slice(0, 800) || "Not provided."}

${contentSection}

[CONTENT METADATA]
Type: ${submission.inputType || "text"} | Source Model: ${submission.sourceModelLabel || "Unknown AI"}

${policyInstruction}

${themeBlock}

[YOUR RULES]
1. Ask EXACTLY ONE focused question per turn — never multiple questions at once.
2. Reference the user's LAST answer explicitly to show you're listening.
3. Reference specific content from the submission when relevant (e.g. "you mentioned the code logic..." or "looking at the image...").
4. If the user is frustrated, acknowledge it warmly before probing.
5. If the user's response is vague, ask ONE clarifying follow-up.
6. Keep questions short (max 2 sentences).
7. Maintain professional but conversational tone.

[RESPONSE FORMAT]
Return ONLY valid JSON — no markdown, no preamble:
{ "reply": "<your single follow-up question>", "shouldEnd": false, "topicCovered": "<one of: first_impression|strengths|weaknesses|accuracy|completeness|relevance|clarity|tone|formatting|hallucination|usability|reasoning|improvement_request|other>" }

If the user wants to end the session, set "shouldEnd": true and write a warm closing message in "reply".`;
}

function policyToInstruction(policy) {
  const map = {
    greet_opening: "Welcome the user warmly and ask for their very first impression of the AI output.",
    ask_first_impression: "Ask how well the generated output answered their original prompt.",
    probe_strength: "Ask what part of the generated output felt strongest or most useful to them.",
    probe_weakness: "Ask where the generated output fell short — probe a specific gap or mismatch.",
    probe_specific_issue: "The user mentioned an issue — ask them to describe it in concrete detail.",
    ask_improvement_priority: "Ask what single improvement would make this output most valuable to them.",
    redirect_to_feedback: "Politely redirect the user back to evaluating the generated content.",
    shorten_question: "The user seems fatigued — ask one brief, easy question to keep momentum.",
    empathy_then_probe: "Acknowledge the user's frustration warmly, then ask one gentle follow-up.",
    confirm_end: "Confirm the user wants to end the session before closing.",
    wrap_up: "Thank the user and signal the session is complete.",
  };
  return map[policy] || "Ask a thoughtful follow-up question about the user's experience with the generated content.";
}

// ---------------------------------------------------------------------------
// Local fallback (when no API key is set)
// ---------------------------------------------------------------------------

function analyzeIntentLocally(userMessage) {
  const message = userMessage.toLowerCase();
  const stopPatterns = ["stop", "end", "done", "enough", "quit", "wrap up"];
  const hesitantPatterns = ["not sure", "maybe later", "short", "tired", "busy"];
  const frustratedPatterns = ["annoyed", "frustrated", "bad", "upset", "hate", "waste"];
  const offTopicPatterns = ["weather", "movie", "sports", "recipe", "joke"];

  if (stopPatterns.some((p) => message.includes(p))) {
    return { sentimentLabel: "wants_to_stop", continueSignal: "stop", offTopic: false, themes: [] };
  }
  if (frustratedPatterns.some((p) => message.includes(p))) {
    return { sentimentLabel: "frustrated", continueSignal: "uncertain", offTopic: false, themes: [] };
  }
  if (hesitantPatterns.some((p) => message.includes(p))) {
    return { sentimentLabel: "hesitant", continueSignal: "uncertain", offTopic: false, themes: [] };
  }
  if (offTopicPatterns.some((p) => message.includes(p))) {
    return { sentimentLabel: "neutral", continueSignal: "continue", offTopic: true, themes: [] };
  }
  return { sentimentLabel: "engaged", continueSignal: "continue", offTopic: false, themes: [] };
}

function buildFallbackQuestion(submission, analysis, questionNumber, coveredThemes) {
  if (analysis.continueSignal === "stop") {
    return {
      reply: `Thank you for your time. I'll close the session and prepare a summary of your feedback on "${submission.title}".`,
      shouldEnd: true,
      topicCovered: "wrap_up",
    };
  }

  if (analysis.offTopic) {
    return {
      reply: `I'd like to keep us focused on your feedback. How well did the generated ${submission.inputType || "content"} match what you originally asked for?`,
      shouldEnd: false,
      topicCovered: "other",
    };
  }

  const prompts = [
    { q: `What was your first impression of how well the generated output answered your original prompt?`, t: "first_impression" },
    { q: `What part of the generated output felt strongest or most useful?`, t: "strengths" },
    { q: `Where did the generated output fall short of your expectations?`, t: "weaknesses" },
    { q: `If you could improve one thing about this output, what would you change first?`, t: "improvement_request" },
  ];

  const available = prompts.filter((p) => !coveredThemes.includes(p.t));
  const chosen = available[(questionNumber - 1) % Math.max(available.length, 1)] || prompts[prompts.length - 1];

  return { reply: chosen.q, shouldEnd: false, topicCovered: chosen.t };
}

// ---------------------------------------------------------------------------
// Summary builder (local fallback)
// ---------------------------------------------------------------------------

function buildSummary(submission, messages) {
  const userMessages = messages.filter((m) => m.role === "user");
  const assistantMessages = messages.filter((m) => m.role === "assistant");
  const sentiments = userMessages.map((m) => m.sentimentLabel).filter(Boolean);
  const combinedFeedback = userMessages.map((m) => m.content).join(" ").toLowerCase();
  const lastUserMessage = userMessages.at(-1)?.content || "";

  const strengths = [];
  const weaknesses = [];
  const recommendations = [];

  if (/(good|great|strong|helpful|clear|coherent|accurate)/.test(combinedFeedback)) {
    strengths.push("User highlighted positive aspects in the generated output, especially around quality or usefulness.");
  } else {
    strengths.push("Session did not surface a major strength with high confidence.");
  }

  if (/(bad|weak|wrong|halluc|unclear|inaccurate|missed|slow)/.test(combinedFeedback)) {
    weaknesses.push("User identified quality gaps or mismatches between the prompt and generated output.");
  } else {
    weaknesses.push("No critical weakness was stated explicitly.");
  }

  if (/(improve|better|change|fix|should|need)/.test(combinedFeedback) || lastUserMessage) {
    recommendations.push("Prioritize the improvement request most clearly expressed by the user during the session.");
  } else {
    recommendations.push("Collect one more direct improvement suggestion in future sessions.");
  }

  return {
    shortSummary: `Feedback session for "${submission.title}" captured ${userMessages.length} user response(s) covering the generated ${submission.inputType || "content"}.`,
    strengths,
    weaknesses,
    recommendations,
    sentimentTimeline: userMessages.map((m, i) => ({ step: i + 1, sentiment: m.sentimentLabel || "neutral" })),
    engagementLevel: sentiments.includes("wants_to_stop") ? "low" : sentiments.includes("hesitant") ? "medium" : "high",
    summaryConfidence: assistantMessages.length > 0 && userMessages.length > 0 ? "medium" : "low",
    overallSentiment: sentiments.at(-1) || "neutral",
    continueSignalFinal: sentiments.includes("wants_to_stop") ? "stop" : "continue",
  };
}

// ---------------------------------------------------------------------------
// Core Gemini caller (native SDK, multi-turn aware, multimodal)
// ---------------------------------------------------------------------------

/**
 * Sends a structured multi-turn request to Gemini.
 *
 * @param {object} opts
 * @param {string}  opts.systemPrompt  - System instruction (grounds Gemini in context)
 * @param {Array}   opts.history       - Prior Gemini contents[] turns
 * @param {Array}   opts.userParts     - Parts array for the current user turn (text + optional inlineData)
 * @returns {Promise<object>} - Parsed JSON from Gemini
 */
async function callGemini({ systemPrompt, history, userParts }) {
  const genAI = getGeminiClient();
  const modelName = env.geminiModel || "gemini-2.0-flash";

  const model = genAI.getGenerativeModel({
    model: modelName,
    systemInstruction: systemPrompt,
    generationConfig: {
      temperature: 0.5,
      responseMimeType: "application/json",
    },
  });

  const chat = model.startChat({ history });

  const result = await chat.sendMessage(userParts);
  const text = result.response.text();

  try {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start === -1 || end === -1) throw new Error("No JSON in Gemini response");
    return JSON.parse(text.slice(start, end + 1));
  } catch {
    throw new HttpError(502, "Gemini returned non-JSON output", text);
  }
}

// ---------------------------------------------------------------------------
// Exported: analyzeUserTurn
// ---------------------------------------------------------------------------

export async function analyzeUserTurn({ message, submission, recentContext, sessionStage }) {
  // ML mode: use in-house model for intent/sentiment classification
  if (env.llmProviderMode === "ml") {
    try {
      const result = await callMlServer("/analyze-turn", {
        input_type: submission?.inputType ?? "text",
        source_model_label: submission?.sourceModelLabel ?? "unknown",
        session_stage: sessionStage ?? "weaknesses",
        original_prompt: submission?.originalPrompt ?? "",
        generated_content: submission?.generatedContent ?? "",
        recent_context: (recentContext ?? []).map((m) => ({ role: m.role, text: m.content })),
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
      console.warn("[llm.provider] ML server unreachable, falling back to local analysis:", err.message);
      return analyzeIntentLocally(message);
    }
  }

  // API mode: ask Gemini to classify the turn
  if (env.llmProviderMode === "api") {
    try {
      const classifyPrompt = `You are a feedback session analyst. Analyze the user's message below in the context of a feedback interview about AI-generated content.

Return ONLY valid JSON with these exact fields:
{
  "sentimentLabel": "<positive|neutral|hesitant|frustrated|wants_to_stop>",
  "continueSignal": "<continue|uncertain|stop>",
  "offTopic": <true|false>,
  "themes": ["<accuracy|completeness|relevance|clarity|tone|formatting|hallucination|latency|usability|reasoning|instruction_following|safety|other>"],
  "feedbackQuality": "<vague|somewhat_actionable|highly_actionable>"
}

Session stage: ${sessionStage || "unknown"}
Original prompt reviewed: "${submission?.originalPrompt?.slice(0, 300) || "N/A"}"
User's message: "${message}"`;

      const genAI = getGeminiClient();
      const model = genAI.getGenerativeModel({
        model: env.geminiModel || "gemini-2.0-flash",
        generationConfig: { temperature: 0.1, responseMimeType: "application/json" },
      });
      const result = await model.generateContent(classifyPrompt);
      const text = result.response.text();
      const start = text.indexOf("{");
      const end = text.lastIndexOf("}");
      return JSON.parse(text.slice(start, end + 1));
    } catch (err) {
      console.warn("[llm.provider] Gemini analysis failed, falling back to local:", err.message);
      return analyzeIntentLocally(message);
    }
  }

  // Mock mode
  return analyzeIntentLocally(message);
}

// ---------------------------------------------------------------------------
// Exported: generateInterviewerReply
// ---------------------------------------------------------------------------

export async function generateInterviewerReply({
  submission,
  userMessage,
  analysis,
  questionNumber,
  recentContext,        // full message history (all turns)
  sessionStage,
  coveredThemes = [],
}) {
  // ---- Determine policy ----
  let policy = null;

  if (env.llmProviderMode === "ml") {
    try {
      const signals = {
        sentiment: analysis.sentimentLabel ?? "neutral",
        topic: analysis.offTopic ? "off_topic" : "on_topic",
        continue_signal: analysis.continueSignal ?? "continue",
        themes: analysis.themes ?? ["other"],
        feedback_quality: analysis.feedbackQuality ?? "somewhat_actionable",
      };
      const result = await callMlServer("/select-policy", {
        session_stage: sessionStage ?? "weaknesses",
        recent_context: (recentContext ?? []).map((m) => ({ role: m.role, text: m.content })),
        model_signals: signals,
      });
      policy = result.policy ?? null;
    } catch (err) {
      console.warn("[llm.provider] ML policy server unreachable:", err.message);
    }
  }

  // Hard stop — always honour regardless of mode
  if (analysis.continueSignal === "stop") {
    return {
      reply: `Thank you so much for your time and feedback on "${submission.title}". I'll now prepare a summary of your session.`,
      shouldEnd: true,
      topicCovered: "wrap_up",
    };
  }

  // ---- Gemini mode (API or ML with Gemini for generation) ----
  if (env.llmProviderMode === "api" || env.llmProviderMode === "ml") {
    try {
      const systemPrompt = buildContextualSystemPrompt(submission, coveredThemes, policy);

      // Build conversation history (all prior turns for full context)
      const history = buildGeminiHistory(
        (recentContext ?? []).slice(0, -1) // exclude the very last message (we send it as current turn)
      );

      // Build current user turn — may include a file part (Phase 2)
      const filePart = await buildFilePart(submission);
      const userParts = [];

      // Inject the file once (only on the first turn so we don't re-send every message)
      if (filePart && questionNumber <= 2) {
        userParts.push(filePart);
      }

      userParts.push({ text: userMessage });

      const geminiResponse = await callGemini({ systemPrompt, history, userParts });

      return {
        reply: geminiResponse.reply || "Thank you for sharing that. Could you tell me more?",
        shouldEnd: Boolean(geminiResponse.shouldEnd),
        topicCovered: geminiResponse.topicCovered || "other",
      };
    } catch (err) {
      console.warn("[llm.provider] Gemini reply generation failed, using fallback:", err.message);
      return buildFallbackQuestion(submission, analysis, questionNumber, coveredThemes);
    }
  }

  // Mock mode
  return buildFallbackQuestion(submission, analysis, questionNumber, coveredThemes);
}

// ---------------------------------------------------------------------------
// Exported: generateSessionSummary
// ---------------------------------------------------------------------------

export async function generateSessionSummary({ submission, messages }) {
  if (env.llmProviderMode === "api" || env.llmProviderMode === "ml") {
    try {
      const userMessages = messages.filter((m) => m.role === "user");
      const conversationText = messages
        .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
        .join("\n");

      const summaryPrompt = `You are a feedback analyst. Based on the interview transcript below, produce a structured summary.

[SUBMISSION]
Title: ${submission.title}
Original Prompt: ${submission.originalPrompt?.slice(0, 500)}
Content Type: ${submission.inputType} | Source Model: ${submission.sourceModelLabel}

[INTERVIEW TRANSCRIPT]
${conversationText.slice(0, 3000)}

Return ONLY valid JSON with these exact fields:
{
  "shortSummary": "<2-3 sentence summary of the feedback session>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "recommendations": ["<recommendation 1>", "<recommendation 2>"],
  "sentimentTimeline": [{"step": 1, "sentiment": "<label>"}],
  "engagementLevel": "<low|medium|high>",
  "summaryConfidence": "<low|medium|high>",
  "overallSentiment": "<positive|neutral|hesitant|frustrated|wants_to_stop>",
  "continueSignalFinal": "<continue|stop>"
}`;

      const genAI = getGeminiClient();
      const model = genAI.getGenerativeModel({
        model: env.geminiModel || "gemini-2.0-flash",
        generationConfig: { temperature: 0.3, responseMimeType: "application/json" },
      });
      const result = await model.generateContent(summaryPrompt);
      const text = result.response.text();
      const start = text.indexOf("{");
      const end = text.lastIndexOf("}");
      return JSON.parse(text.slice(start, end + 1));
    } catch (err) {
      console.warn("[llm.provider] Gemini summary failed, using local builder:", err.message);
      return buildSummary(submission, messages);
    }
  }

  return buildSummary(submission, messages);
}
