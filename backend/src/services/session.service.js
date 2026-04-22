import prismaPackage from "@prisma/client";

import { prisma } from "../config/prisma.js";
import { generateInterviewerReply, generateSessionSummary, analyzeUserTurn } from "./llm.provider.js";
import { HttpError } from "../lib/http-error.js";

const { SessionStatus } = prismaPackage;

// ---------------------------------------------------------------------------
// Derive the themes already covered from prior assistant messages.
// We store a topicCovered marker on each assistant message so we can
// reconstruct the set without re-parsing text.
// ---------------------------------------------------------------------------

function deriveCoveredThemes(messages) {
  return messages
    .filter((m) => m.role === "assistant" && m.topicCovered)
    .map((m) => m.topicCovered)
    .filter(Boolean);
}

// ---------------------------------------------------------------------------
// Create a new interview session with a contextual opening message
// ---------------------------------------------------------------------------

export async function createInterviewSession({ submissionId, userId }) {
  const submission = await prisma.submission.findFirst({
    where: {
      id: submissionId,
      userId,
    },
  });

  if (!submission) {
    throw new HttpError(404, "Submission not found");
  }

  // Build a contextual opening that references the actual submission
  const inputLabel =
    submission.inputType === "image" ? "image"
    : submission.inputType === "audio" ? "audio recording"
    : submission.inputType === "video" ? "video"
    : submission.inputType === "pdf" ? "document"
    : submission.inputType === "code" ? "code output"
    : "response";

  const openingMessage = `Hello, and welcome to this feedback session about "${submission.title}". I'm here to collect your thoughts on the AI-generated ${inputLabel} you submitted. To start — what was your very first impression when you saw the output?`;

  const session = await prisma.interviewSession.create({
    data: {
      submissionId,
      userId,
      status: SessionStatus.ACTIVE,
      messages: {
        create: {
          role: "assistant",
          content: openingMessage,
          topicCovered: "first_impression",
        },
      },
    },
    include: {
      messages: {
        orderBy: { createdAt: "asc" },
      },
      submission: true,
    },
  });

  return session;
}

// ---------------------------------------------------------------------------
// Process a new user turn
// ---------------------------------------------------------------------------

export async function addSessionMessage({ sessionId, userId, content }) {
  const session = await prisma.interviewSession.findFirst({
    where: {
      id: sessionId,
      userId,
    },
    include: {
      submission: true,
      messages: {
        orderBy: { createdAt: "asc" },
      },
      summary: true,
    },
  });

  if (!session) {
    throw new HttpError(404, "Session not found");
  }

  if (session.status !== SessionStatus.ACTIVE) {
    throw new HttpError(400, "This session is already closed");
  }

  // Determine the current stage from the last assistant message
  const currentStage = session.messages.length > 0
    ? (session.messages[session.messages.length - 1]?.sessionStage ?? "weaknesses")
    : "first_impression";

  // Derive what topics have already been covered (Phase 1/2 context)
  const coveredThemes = deriveCoveredThemes(session.messages);

  // Classify the user's turn
  // Pass full recent context (not just 4 messages) so ML model has complete picture
  const analysis = await analyzeUserTurn({
    message: content,
    submission: session.submission,
    recentContext: session.messages,        // full history
    sessionStage: currentStage,
  });

  // Persist user message
  const userMessage = await prisma.sessionMessage.create({
    data: {
      sessionId: session.id,
      role: "user",
      content,
      sentimentLabel: analysis.sentimentLabel,
      offTopicFlag: Boolean(analysis.offTopic),
    },
  });

  const userQuestionCount = session.messages.filter((m) => m.role === "user").length + 1;

  // Generate the next contextual interviewer question
  // Pass ALL messages so Gemini has the full multi-turn conversation context
  const interviewer = await generateInterviewerReply({
    submission: session.submission,
    userMessage: content,
    analysis,
    questionNumber: userQuestionCount + 1,
    recentContext: [
      ...session.messages,
      { role: "user", content },        // include the just-submitted message
    ],
    sessionStage: currentStage,
    coveredThemes,                       // so Gemini doesn't repeat topics
  });

  // Persist assistant message with topicCovered for future context building
  const assistantMessage = await prisma.sessionMessage.create({
    data: {
      sessionId: session.id,
      role: "assistant",
      content: interviewer.reply,
      topicCovered: interviewer.topicCovered ?? null,
    },
  });

  const completionScore = Math.min(userQuestionCount * 25, 100);

  await prisma.interviewSession.update({
    where: { id: session.id },
    data: {
      completionScore,
      overallSentiment: analysis.sentimentLabel,
      continueSignalFinal: analysis.continueSignal,
    },
  });

  // End session only when the LLM determines it has enough actionable feedback or if the user explicitly wants to stop
  if (interviewer.shouldEnd) {
    const endedBy = "SYSTEM";
    const endReason = "LLM indicated interview is complete or user wanted to stop";
    await endInterviewSession({ sessionId: session.id, userId, endedBy, endReason });
  }

  const refreshed = await prisma.interviewSession.findUnique({
    where: { id: session.id },
    include: {
      messages: {
        orderBy: { createdAt: "asc" },
      },
      summary: true,
    },
  });

  return {
    session: refreshed,
    latestMessages: [userMessage, assistantMessage],
  };
}

// ---------------------------------------------------------------------------
// End the session and generate a summary
// ---------------------------------------------------------------------------

export async function endInterviewSession({ sessionId, userId, endedBy = "USER", endReason = "Session ended manually" }) {
  const session = await prisma.interviewSession.findFirst({
    where: {
      id: sessionId,
      userId,
    },
    include: {
      submission: true,
      messages: {
        orderBy: { createdAt: "asc" },
      },
      summary: true,
    },
  });

  if (!session) {
    throw new HttpError(404, "Session not found");
  }

  if (session.summary) {
    return session.summary;
  }

  // Generate a contextually rich summary using the full conversation
  const summaryPayload = await generateSessionSummary({
    submission: session.submission,
    messages: session.messages,
  });

  const status = endedBy === "SYSTEM" ? SessionStatus.COMPLETED : SessionStatus.ENDED_EARLY;

  const result = await prisma.$transaction(async (tx) => {
    await tx.interviewSession.update({
      where: { id: session.id },
      data: {
        status,
        endedAt: new Date(),
        endedBy,
        endReason,
        overallSentiment: summaryPayload.overallSentiment,
        continueSignalFinal: summaryPayload.continueSignalFinal,
      },
    });

    const summary = await tx.sessionSummary.create({
      data: {
        sessionId: session.id,
        shortSummary: summaryPayload.shortSummary,
        strengths: summaryPayload.strengths,
        weaknesses: summaryPayload.weaknesses,
        recommendations: summaryPayload.recommendations,
        sentimentTimeline: summaryPayload.sentimentTimeline,
        engagementLevel: summaryPayload.engagementLevel,
        summaryConfidence: summaryPayload.summaryConfidence,
      },
    });

    await tx.sessionMessage.deleteMany({
      where: {
        sessionId: session.id,
      },
    });

    return summary;
  });

  return result;
}
