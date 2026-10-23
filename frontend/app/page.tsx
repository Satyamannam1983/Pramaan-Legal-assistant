"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import { AppShell } from "../components/AppShell";
import { PageHeader } from "../components/PageHeader";
import { buildFollowUpPrompts } from "../lib/follow-up";
import {
  askDocument,
  askQuestion,
  compareDocuments,
  getErrorMessage,
  listDocuments,
  safeSourceUrl,
  uploadDocument,
  type ChatResponse,
  type ComparisonCitation,
  type DocumentComparisonResponse,
  type DocumentSummary,
  type SearchResult,
  type CitationRef,
  type UploadResponse,
} from "../lib/api";

type WorkspaceMode = "general" | "document" | "compare";
type Citation = CitationRef | ComparisonCitation;
type ChatTurn = {
  id: string;
  question: string;
  answer: string;
  citations: Citation[];
  sources: SearchResult[];
  abstained: boolean;
  limitations: string[];
  mode: WorkspaceMode;
};

function makeComparisonAnswer(result: DocumentComparisonResponse): string {
  const sections = [
    result.overview,
    result.similarities.length ? `Similarities\n${result.similarities.map((item) => `- ${item}`).join("\n")}` : "",
    result.differences.length ? `Differences\n${result.differences.map((item) => `- ${item}`).join("\n")}` : "",
    result.legal_provisions.length ? `Legal provisions\n${result.legal_provisions.map((item) => `- ${item}`).join("\n")}` : "",
    result.reasoning_comparison.length ? `Reasoning comparison\n${result.reasoning_comparison.map((item) => `- ${item}`).join("\n")}` : "",
    result.outcome_comparison ? `Outcome\n${result.outcome_comparison}` : "",
  ];
  return sections.filter(Boolean).join("\n\n");
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

function CitationIcon() {
  return (
    <svg className="citation-open-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M14 4h6v6M20 4l-9 9" />
      <path d="M18 13v5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h5" />
    </svg>
  );
}

function citationTitle(citation: Citation, documents: DocumentSummary[]) {
  if ("title" in citation && citation.title) return citation.title;
  return documents.find((doc) => doc.document_id === citation.document_id)?.title
    || citation.document_id;
}

function citationHref(citation: Citation) {
  if ("source_url" in citation) return safeSourceUrl(citation.source_url);
  return null;
}

function citationsDestination(citation: Citation) {
  return `/documents/${encodeURIComponent(citation.document_id)}`;
}

export default function Home() {
  const [mode, setMode] = useState<WorkspaceMode>("general");
  const [query, setQuery] = useState("");
  const [selectedDocument, setSelectedDocument] = useState("");
  const [comparisonDocuments, setComparisonDocuments] = useState<[string, string]>(["", ""]);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [documentsError, setDocumentsError] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeCitationTab, setActiveCitationTab] = useState<"citations" | "sources">("citations");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [attachedFile, setAttachedFile] = useState<File | null>(null);
  const requestController = useRef<AbortController | null>(null);
  const uploadController = useRef<AbortController | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const questionInput = useRef<HTMLTextAreaElement | null>(null);
  const chatScroll = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    listDocuments(controller.signal)
      .then((items) => {
        if (!controller.signal.aborted) {
          setDocuments(items);
          setSelectedDocument(items[0]?.document_id ?? "");
        }
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted) setDocumentsError(getErrorMessage(failure));
      })
      .finally(() => {
        if (!controller.signal.aborted) setDocumentsLoading(false);
      });
    return () => {
      controller.abort();
      requestController.current?.abort();
      uploadController.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (chatScroll.current) chatScroll.current.scrollTop = chatScroll.current.scrollHeight;
  }, [turns, pendingQuestion]);

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || loading || mode === "compare") return;
    if (mode === "document" && !selectedDocument) {
      setError("Choose an indexed document before asking a document-specific question.");
      return;
    }

    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    setLoading(true);
    setPendingQuestion(trimmed);
    setError("");

    try {
      const response: ChatResponse = mode === "document"
        ? await askDocument(selectedDocument, { query: trimmed }, controller.signal)
        : await askQuestion({ query: trimmed, conversation_id: conversationId }, controller.signal);
      if (requestController.current !== controller) return;
      setQuery("");
      if (mode === "general") setConversationId(response.conversation_id);
      setTurns((current) => [
        ...current,
        {
          id: `${response.conversation_id}:${current.length}`,
          question: trimmed,
          answer: response.answer,
          citations: response.citations,
          sources: response.search_results,
          abstained: response.abstained,
          limitations: response.limitations,
          mode,
        },
      ]);
      setActiveCitationTab("citations");
    } catch (failure) {
      if (!controller.signal.aborted && requestController.current === controller) {
        setError(getErrorMessage(failure));
      }
    } finally {
      if (requestController.current === controller) {
        requestController.current = null;
        setPendingQuestion("");
        setLoading(false);
      }
    }
  }

  async function runComparison() {
    const [first, second] = comparisonDocuments;
    if (!first || !second || first === second || loading) return;
    const firstTitle = documents.find((doc) => doc.document_id === first)?.title || first;
    const secondTitle = documents.find((doc) => doc.document_id === second)?.title || second;
    const controller = new AbortController();
    requestController.current?.abort();
    requestController.current = controller;
    setLoading(true);
    setPendingQuestion(`Compare ${firstTitle} and ${secondTitle}`);
    setError("");
    try {
      const result = await compareDocuments({ document_ids: [first, second] }, controller.signal);
      if (requestController.current !== controller) return;
      setTurns((current) => [
        ...current,
        {
          id: `comparison:${current.length}`,
          question: `Compare ${firstTitle} and ${secondTitle}`,
          answer: makeComparisonAnswer(result),
          citations: result.citations,
          sources: [],
          abstained: false,
          limitations: result.limitations,
          mode: "compare",
        },
      ]);
      setActiveCitationTab("citations");
    } catch (failure) {
      if (!controller.signal.aborted && requestController.current === controller) {
        setError(getErrorMessage(failure));
      }
    } finally {
      if (requestController.current === controller) {
        requestController.current = null;
        setPendingQuestion("");
        setLoading(false);
      }
    }
  }

  async function uploadAttachment(file: File) {
    setAttachedFile(file);
    setUploadMessage("");
    setUploadError("");
    setUploadResult(null);
    const controller = new AbortController();
    uploadController.current?.abort();
    uploadController.current = controller;
    setUploading(true);
    try {
      const result = await uploadDocument(file, controller.signal);
      if (uploadController.current !== controller) return;
      setUploadResult(result);
      if (result.indexing_status === "indexed") {
        setUploadMessage(`${result.filename} is uploaded and indexed. Ask your question about it below.`);
        if (result.document_id) {
          setSelectedDocument(result.document_id);
          setMode("document");
        }
        try {
          const refreshed = await listDocuments(controller.signal);
          if (uploadController.current === controller) setDocuments(refreshed);
        } catch {
          if (!controller.signal.aborted && uploadController.current === controller) {
            setUploadMessage(`${result.filename} was uploaded and indexed, but the document list could not be refreshed.`);
          }
        }
      } else {
        setUploadMessage(`${result.filename} was uploaded, but indexing is ${result.indexing_status}. It may not be available for retrieval.`);
      }
      setAttachedFile(null);
      if (fileInput.current) fileInput.current.value = "";
    } catch (failure) {
      if (!controller.signal.aborted && uploadController.current === controller) {
        setUploadError(getErrorMessage(failure));
      }
    } finally {
      if (uploadController.current === controller) {
        uploadController.current = null;
        setUploading(false);
      }
    }
  }

  const latest = turns[turns.length - 1];
  const latestCitations = latest?.citations ?? [];
  const latestSources = latest?.sources ?? [];
  const citationMode = activeCitationTab === "citations";

  function changeMode(nextMode: WorkspaceMode) {
    setMode(nextMode);
    setError("");
  }

  function useFollowUp(prompt: { documentId: string; query: string }) {
    setSelectedDocument(prompt.documentId);
    setMode("document");
    setQuery(prompt.query);
    setError("");
    requestAnimationFrame(() => questionInput.current?.focus());
  }

  return (
    <AppShell>
      <main className="page-container assistant-page">
        <PageHeader
          eyebrow="Legal Assistant"
          title="Legal Assistant"
          description="Ask legal questions, analyze documents, and get AI-powered answers with verifiable citations."
        />

        <div className="chat-grid">
          <section className="surface-panel chat-panel" aria-label="Legal assistant conversation">
            <div className="chat-modebar" role="tablist" aria-label="Question type">
              <button type="button" role="tab" aria-selected={mode === "general"} onClick={() => changeMode("general")} className={`mode-button${mode === "general" ? " mode-button-active" : ""}`}>
                <svg className="mode-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M5 4h14a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-7l-5 3v-3H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z" /><path d="M8 9h8M8 13h5" /></svg>
                General Question
              </button>
              <button type="button" role="tab" aria-selected={mode === "document"} onClick={() => changeMode("document")} className={`mode-button${mode === "document" ? " mode-button-active" : ""}`}>
                <svg className="mode-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M6 3h8l4 4v14H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Z" /><path d="M14 3v5h5M8 13h8M8 17h6" /></svg>
                Ask About Document
              </button>
              <button type="button" role="tab" aria-selected={mode === "compare"} onClick={() => changeMode("compare")} className={`mode-button${mode === "compare" ? " mode-button-active" : ""}`}>
                <svg className="mode-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M12 3v17M7 21h10M5 6h14M5 6l-3 7h6L5 6ZM19 6l-3 7h6l-3-7Z" /><path d="M2 13a3 3 0 0 0 6 0M16 13a3 3 0 0 0 6 0" /></svg>
                Compare Documents
              </button>
            </div>

            {mode !== "general" && (
              <div className="chat-scope-controls">
                {mode === "document" ? (
                  <>
                    <label className="sr-only" htmlFor="document-scope">Choose an indexed document</label>
                    <select id="document-scope" className="select-input" value={selectedDocument} onChange={(event) => setSelectedDocument(event.target.value)} disabled={documentsLoading || loading}>
                      <option value="">Choose an indexed document</option>
                      {documents.map((doc) => <option key={doc.document_id} value={doc.document_id}>{doc.title || doc.document_id}</option>)}
                    </select>
                    <p className="helper-text" role={documentsError ? "alert" : undefined}>
                      {documentsError || (documentsLoading ? "Loading indexed documents..." : `${documents.length} indexed document${documents.length === 1 ? "" : "s"} available`)}
                    </p>
                  </>
                ) : (
                  <>
                    {([0, 1] as const).map((index) => (
                      <div className="scope-select-field" key={index}>
                        <label className="helper-text" htmlFor={`comparison-document-${index}`}>
                          {index === 0 ? "First document" : "Second document"}
                        </label>
                        <select
                          id={`comparison-document-${index}`}
                          className="select-input"
                          value={comparisonDocuments[index]}
                          onChange={(event) => setComparisonDocuments((current) => index === 0 ? [event.target.value, current[1]] : [current[0], event.target.value])}
                          disabled={documentsLoading || loading}
                        >
                          <option value="">{index === 0 ? "Select first document" : "Select second document"}</option>
                          {documents.map((doc) => <option key={doc.document_id} value={doc.document_id}>{doc.title || doc.document_id}</option>)}
                        </select>
                      </div>
                    ))}
                    <div className="col-span-full flex flex-wrap items-center justify-between gap-2">
                      <p className="helper-text">{documentsError || (documentsLoading ? "Loading indexed documents..." : "Choose two different indexed documents to compare.")}</p>
                      <button type="button" className="neutral-button" onClick={runComparison} disabled={documentsLoading || loading || !comparisonDocuments[0] || !comparisonDocuments[1] || comparisonDocuments[0] === comparisonDocuments[1]}>
                        {loading ? "Comparing..." : "Compare selected documents"}
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}

            <div className="chat-scroll" ref={chatScroll} aria-live="polite">
              {!turns.length && !pendingQuestion && (
                <div className="chat-empty">
                  <span className="chat-empty-mark"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M12 3v17M7 21h10M5 6h14M5 6l-3 7h6L5 6ZM19 6l-3 7h6l-3-7Z" /><path d="M2 13a3 3 0 0 0 6 0M16 13a3 3 0 0 0 6 0" /></svg></span>
                  <h2>Research with verifiable sources</h2>
                  <p>Submit a question to retrieve evidence from indexed legal documents. Citations and related passages will appear alongside the answer.</p>
                </div>
              )}
              {turns.map((turn) => {
                const followUps = buildFollowUpPrompts(
                  turn.citations.length ? turn.citations : turn.sources,
                );
                return <article className="chat-turn" key={turn.id}>
                  <div className="user-message-row">
                    <div className="user-message">{turn.question}</div>
                  </div>
                  <div className="assistant-message-row">
                    <AssistantMark />
                    <div className="assistant-message">
                      <div className="assistant-message-meta">
                        <span>Pramaan</span>
                        <span>{turn.mode === "compare" ? "Document comparison" : turn.mode === "document" ? "Document-specific research" : "Legal research"}</span>
                      </div>
                      <AnswerContent answer={turn.answer} />
                      {turn.abstained && <p className="insufficient-note">The system abstained because the available evidence was not sufficient to support a verified answer.</p>}
                      {!turn.citations.length && !turn.sources.length && !turn.abstained && (
                        <p className="insufficient-note">No citations or supporting passages were returned for this response.</p>
                      )}
                      {!!turn.limitations.length && <p className="message-limitations">{turn.limitations.join(" ")}</p>}
                      {followUps.length > 0 && (
                        <div className="follow-up-prompts" aria-label="Evidence-grounded follow-up questions">
                          <p className="follow-up-heading">Ask about the cited document</p>
                          {followUps.map((prompt) => (
                            <button
                              type="button"
                              key={`${prompt.documentId}:${prompt.label}`}
                              className="follow-up-button"
                              onClick={() => useFollowUp(prompt)}
                              disabled={loading}
                            >
                              {prompt.label}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </article>
              })}
              {pendingQuestion && (
                <article className="chat-turn">
                  <div className="user-message-row"><div className="user-message">{pendingQuestion}</div></div>
                  <div className="pending-message"><AssistantMark /><span>Retrieving evidence and preparing a response</span><span className="loading-dots" aria-label="Loading"><span /><span /><span /></span></div>
                </article>
              )}
            </div>
          </section>

          <aside className="surface-panel citation-panel" aria-label="Sources for the latest response">
            <div className="citation-tabs" role="tablist" aria-label="Evidence panel">
              <button type="button" role="tab" aria-selected={citationMode} onClick={() => setActiveCitationTab("citations")} className={`citation-tab${citationMode ? " citation-tab-active" : ""}`}>
                Citations ({latestCitations.length})
              </button>
              <button type="button" role="tab" aria-selected={!citationMode} onClick={() => setActiveCitationTab("sources")} className={`citation-tab${!citationMode ? " citation-tab-active" : ""}`}>
                Related Sources ({latestSources.length})
              </button>
            </div>
            <div className="citation-list">
              {citationMode ? (
                latestCitations.length ? latestCitations.map((citation, index) => {
                  const href = citationHref(citation);
                  const title = citationTitle(citation, documents);
                  const contents = (
                    <>
                      <span className="citation-number">{index + 1}</span>
                      <span className="citation-copy">
                        <span className="citation-title">{title}</span>
                        {citation.section && <span className="citation-detail">{citation.section}{"subsection" in citation && citation.subsection ? ` · ${citation.subsection}` : ""}</span>}
                        {citation.page_start != null && <span className="citation-detail">Page {citation.page_start}{citation.page_end != null && citation.page_end !== citation.page_start ? `–${citation.page_end}` : ""}</span>}
                        {citation.relevant_quote && <span className="citation-quote">{citation.relevant_quote}</span>}
                      </span>
                      <CitationIcon />
                    </>
                  );
                  return href
                    ? <a className="citation-row" href={href} target="_blank" rel="noopener noreferrer" key={`${citation.document_id}:${citation.chunk_id}`}>{contents}</a>
                    : <Link className="citation-row" href={citationsDestination(citation)} key={`${citation.document_id}:${citation.chunk_id}`}>{contents}</Link>;
                }) : <div className="citation-empty">Citations returned for the latest answer will appear here.</div>
              ) : (
                latestSources.length ? latestSources.map((source, index) => {
                  const href = safeSourceUrl(source.source_url);
                  const contents = (
                    <>
                      <span className="citation-number">{source.final_rank ?? index + 1}</span>
                      <span className="citation-copy">
                        <span className="citation-title">{source.title || source.document_id}</span>
                        {source.section && <span className="citation-detail">{source.section}{source.subsection ? ` · ${source.subsection}` : ""}</span>}
                        {source.page_start != null && <span className="citation-detail">Page {source.page_start}{source.page_end != null && source.page_end !== source.page_start ? `–${source.page_end}` : ""}</span>}
                        <span className="citation-quote">{source.content.length > 220 ? `${source.content.slice(0, 217)}…` : source.content}</span>
                      </span>
                      <CitationIcon />
                    </>
                  );
                  return href
                    ? <a className="citation-row" href={href} target="_blank" rel="noopener noreferrer" key={source.chunk_id}>{contents}</a>
                    : <Link className="citation-row" href={`/documents/${encodeURIComponent(source.document_id)}`} key={source.chunk_id}>{contents}</Link>;
                }) : <div className="citation-empty">Related passages returned by retrieval will appear here.</div>
              )}
            </div>
          </aside>
        </div>

        <form className="question-composer" onSubmit={submitQuestion}>
          <label className="sr-only" htmlFor="legal-question">Ask a legal question</label>
          <textarea
            ref={questionInput}
            id="legal-question"
            className="question-textarea"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && mode !== "compare") {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            maxLength={4000}
            disabled={loading || mode === "compare"}
            placeholder={mode === "document" ? "Ask a question about the selected document..." : mode === "compare" ? "Select two documents above to run a comparison." : "Ask a legal question..."}
            rows={2}
          />
          <div className="composer-footer">
            <div className="composer-tools">
              <input
                ref={fileInput}
                type="file"
                accept=".pdf,.txt,.html,.htm,.json"
                className="composer-file"
                aria-label="Attach a legal document"
                disabled={uploading}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void uploadAttachment(file);
                }}
              />
              <button type="button" className="composer-tool" onClick={() => fileInput.current?.click()} disabled={uploading}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m20 11.5-7.7 7.7a5 5 0 0 1-7.1-7.1l8.1-8.1a3.3 3.3 0 0 1 4.7 4.7l-8.1 8.1a1.7 1.7 0 0 1-2.4-2.4l7.4-7.4" /></svg>
                {uploading ? "Uploading..." : attachedFile ? attachedFile.name : "Attach Document"}
              </button>
              <span className="composer-divider" />
              <span className="composer-tool composer-tool-static" aria-label={mode === "document" ? "Question scope: selected document" : "Question scope: all indexed documents"}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" /></svg>
                {mode === "document" && selectedDocument ? documents.find((doc) => doc.document_id === selectedDocument)?.title || "Selected document" : "All indexed documents"}
              </span>
            </div>
            <button type="submit" className="dark-button" disabled={loading || mode === "compare" || !query.trim() || (mode === "document" && !selectedDocument)}>
              {loading && mode !== "compare" ? "Sending..." : "Send"}
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" width="15" height="15"><path d="m21 3-7.5 18-3.8-7.7L2 9.5 21 3Z" /><path d="M9.7 13.3 21 3" /></svg>
            </button>
          </div>
          {(uploadMessage || uploadError || (uploadResult && uploadResult.indexing_status !== "indexed")) && (
            <p className={`upload-status${uploadError ? " upload-status-error" : ""}`} role={uploadError ? "alert" : "status"}>
              {uploadError || uploadMessage}
              {uploadResult?.detail ? ` ${uploadResult.detail}` : ""}
            </p>
          )}
        </form>
        {error && <p className="assistant-error" role="alert">{error}</p>}
      </main>
    </AppShell>
  );
}
