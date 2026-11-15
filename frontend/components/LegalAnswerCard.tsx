"use client";

import Link from "next/link";
import { buildFollowUpPrompts, type FollowUpPrompt } from "../lib/follow-up";
import type { ChatTurn } from "../lib/chat-session";

type Props = {
  turn: ChatTurn;
  onFollowUp: (prompt: FollowUpPrompt) => void;
  followUpsDisabled: boolean;
};

function answerStatus(turn: ChatTurn) {
  const limitations = turn.limitations.join(" ").toLowerCase();
  if (turn.abstained && limitations.includes("citation")) {
    return { label: "Could not verify", tone: "error" };
  }
  if (turn.abstained && (limitations.includes("timeout") || limitations.includes("in time"))) {
    return { label: "Generation timed out", tone: "warning" };
  }
  if (turn.abstained && turn.generationMode === "failed") {
    return { label: "Generation unavailable", tone: "warning" };
  }
  if (turn.abstained && limitations.includes("retriev")) {
    return { label: "Sources unavailable", tone: "warning" };
  }
  if (turn.abstained) {
    return { label: "Insufficient evidence", tone: "neutral" };
  }
  if (turn.citations.length > 0) {
    return { label: "Evidence supported", tone: "success" };
  }
  return { label: "No verified citation", tone: "neutral" };
}

function AnswerContent({ answer }: { answer: string }) {
  const lines = answer.split(/\r?\n/);
  const blocks: Array<{ type: "paragraph" | "heading" | "ul" | "ol"; content: string[] }> = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const heading = trimmed.match(/^#{1,3}\s+(.+)$/);
    const unordered = trimmed.match(/^[-*]\s+(.+)$/);
    const ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (heading) {
      blocks.push({ type: "heading", content: [heading[1]] });
    } else if (unordered || ordered) {
      const type = unordered ? "ul" : "ol";
      const value = (unordered ?? ordered)?.[1] ?? trimmed;
      const last = blocks[blocks.length - 1];
      if (last?.type === type) last.content.push(value);
      else blocks.push({ type, content: [value] });
    } else {
      const last = blocks[blocks.length - 1];
      if (last?.type === "paragraph") last.content.push(trimmed);
      else blocks.push({ type: "paragraph", content: [trimmed] });
    }
  }

  return (
    <div className="assistant-answer">
      {blocks.map((block, index) => {
        if (block.type === "heading") return <h3 key={index}>{block.content[0]}</h3>;
        if (block.type === "ul" || block.type === "ol") {
          const List = block.type === "ul" ? "ul" : "ol";
          return <List key={index}>{block.content.map((item, itemIndex) => <li key={itemIndex}>{formatInline(item)}</li>)}</List>;
        }
        return <p key={index}>{formatInline(block.content.join(" "))}</p>;
      })}
    </div>
  );
}

function formatInline(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) =>
    part.startsWith("**") && part.endsWith("**")
      ? <strong key={index}>{part.slice(2, -2)}</strong>
      : part
  );
}

function AssistantMark() {
  return (
    <span className="message-avatar" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 3v17M7 21h10M5 6h14M5 6l-3 7h6L5 6ZM19 6l-3 7h6l-3-7Z" />
        <path d="M2 13a3 3 0 0 0 6 0M16 13a3 3 0 0 0 6 0" />
      </svg>
    </span>
  );
}

export function LegalAnswerCard({ turn, onFollowUp, followUpsDisabled }: Props) {
  const status = answerStatus(turn);
  const followUps = turn.abstained
    ? []
    : buildFollowUpPrompts(
      turn.citations.length ? turn.citations : turn.sources,
      turn.answer,
    );
  const evidence = turn.citations.map((citation) => {
    const source = turn.sources.find(
      (item) => item.document_id === citation.document_id && item.chunk_id === citation.chunk_id,
    );
    const quote = citation.relevant_quote || source?.content || null;
    return { citation, quote };
  });

  return (
    <div className="assistant-message-row">
      <AssistantMark />
      <div className="assistant-message">
        <article className={`answer-card${turn.abstained ? " answer-card-abstained" : ""}`}>
          <header className="answer-card-header">
            <div className="answer-brand">
              <span className="answer-brand-mark" aria-hidden="true">P</span>
              <span>
                <strong>Pramaan</strong>
                <span className="answer-mode">{turn.mode === "compare" ? "Document comparison" : turn.mode === "document" ? "Document research" : "Legal research"}</span>
              </span>
            </div>
            <span className={`answer-status answer-status-${status.tone}`}>
              <span className="answer-status-dot" aria-hidden="true" />
              {status.label}
            </span>
          </header>

          <div className="answer-card-content">
            <AnswerContent answer={turn.answer} />
            {turn.abstained && (
              <p className="answer-abstention" role="status">
                {status.label === "Could not verify"
                  ? "The answer was not accepted because its citations could not be verified against the retrieved sources."
                  : status.label === "Generation timed out"
                    ? "The response took too long to generate. You can retry this question."
                    : status.label === "Generation unavailable"
                      ? "A verified response could not be generated. Please retry this question."
                    : status.label === "Sources unavailable"
                      ? "Source retrieval is unavailable right now. Please try again."
                      : "The available sources do not contain enough information to answer this question reliably."}
              </p>
            )}
            {!!turn.limitations.length && !turn.abstained && (
              <p className="answer-limitations">{turn.limitations.join(" ")}</p>
            )}
          </div>

          {!!evidence.length && (
            <details className="answer-evidence">
              <summary>
                <span>Supporting evidence</span>
                <span className="answer-evidence-count">{evidence.length}</span>
              </summary>
              <div className="answer-evidence-list">
                {evidence.map(({ citation, quote }, index) => {
                  const title = "title" in citation && citation.title
                    ? citation.title
                    : citation.document_id;
                  return (
                    <article className="answer-evidence-item" key={`${citation.document_id}:${citation.chunk_id}`}>
                      <div className="answer-evidence-meta">
                        <span className="citation-number">{index + 1}</span>
                        <div>
                          <strong>{title}</strong>
                          {citation.page_start != null && (
                            <span>Page {citation.page_start}{citation.page_end != null && citation.page_end !== citation.page_start ? `–${citation.page_end}` : ""}</span>
                          )}
                        </div>
                        <Link
                          className="answer-evidence-open"
                          href={`/documents/${encodeURIComponent(citation.document_id)}`}
                          aria-label={`Open ${title}`}
                        >
                          Open
                        </Link>
                      </div>
                      {quote ? <blockquote>{quote}</blockquote> : <p className="answer-evidence-missing">The source passage is not included in this response.</p>}
                    </article>
                  );
                })}
              </div>
            </details>
          )}

          {!!followUps.length && (
            <div className="follow-up-prompts" aria-label="Follow-up actions">
              <span className="follow-up-heading">Continue researching</span>
              {followUps.map((prompt) => (
                <button
                  type="button"
                  key={`${prompt.documentId}:${prompt.label}`}
                  className="follow-up-button"
                  onClick={() => onFollowUp(prompt)}
                  disabled={followUpsDisabled}
                >
                  {prompt.label}
                </button>
              ))}
            </div>
          )}
        </article>
      </div>
    </div>
  );
}
