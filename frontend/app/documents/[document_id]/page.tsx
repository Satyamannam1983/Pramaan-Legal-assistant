"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { AppShell } from "../../../components/AppShell";
import { PageHeader } from "../../../components/PageHeader";
import { buildFollowUpPrompts } from "../../../lib/follow-up";
import {
  askDocument,
  ApiError,
  getDocument,
  getDocumentAnalysis,
  getErrorMessage,
  safeSourceUrl,
  type ChatResponse,
  type DocumentAnalysisResponse,
  type DocumentDetail,
} from "../../../lib/api";

export default function DocumentDetailPage() {
  const params = useParams<{ document_id: string }>();
  const documentId = params.document_id;
  const [document, setDocument] = useState<DocumentDetail | null>(null);
  const [analysis, setAnalysis] = useState<DocumentAnalysisResponse | null>(null);
  const [loadingDocument, setLoadingDocument] = useState(true);
  const [documentError, setDocumentError] = useState("");
  const [analysisError, setAnalysisError] = useState("");
  const [query, setQuery] = useState("");
  const [askedQuery, setAskedQuery] = useState("");
  const [answer, setAnswer] = useState<ChatResponse | null>(null);
  const [asking, setAsking] = useState(false);
  const [questionError, setQuestionError] = useState("");
  const questionController = useRef<AbortController | null>(null);
  const questionInput = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setDocument(null);
    setAnalysis(null);
    setAnswer(null);
    setQuery("");
    setAskedQuery("");
    setAsking(false);
    setQuestionError("");
    setLoadingDocument(true);
    setDocumentError("");
    setAnalysisError("");
    Promise.allSettled([
      getDocument(documentId, controller.signal),
      getDocumentAnalysis(documentId, controller.signal),
    ]).then(([documentResult, analysisResult]) => {
      if (controller.signal.aborted) return;
      if (documentResult.status === "fulfilled") {
        setDocument(documentResult.value);
      } else {
        setDocumentError(getErrorMessage(documentResult.reason));
      }
      if (analysisResult.status === "fulfilled") {
        setAnalysis(analysisResult.value);
      } else {
        setAnalysis(null);
        setAnalysisError(
          analysisResult.reason instanceof ApiError && analysisResult.reason.status === 404
            ? "Analysis is not available yet; this document may not have indexed passages."
            : getErrorMessage(analysisResult.reason),
        );
      }
    }).finally(() => {
      if (!controller.signal.aborted) setLoadingDocument(false);
    });
    return () => {
      controller.abort();
      const activeQuestion = questionController.current;
      questionController.current = null;
      activeQuestion?.abort();
    };
  }, [documentId]);

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || asking) {
      if (!trimmed) setQuestionError("Enter a question about this document.");
      return;
    }
    questionController.current?.abort();
    const controller = new AbortController();
    questionController.current = controller;
    setAsking(true);
    setQuestionError("");
    setAnswer(null);
    setAskedQuery(trimmed);
    try {
      const result = await askDocument(documentId, { query: trimmed }, controller.signal);
      if (questionController.current === controller) setAnswer(result);
    } catch (error) {
      if (!controller.signal.aborted && questionController.current === controller) {
        setQuestionError(getErrorMessage(error));
      }
    } finally {
      if (questionController.current === controller) {
        questionController.current = null;
        setAsking(false);
      }
    }
  }

  function useFollowUp(prompt: { query: string }) {
    setQuery(prompt.query);
    setQuestionError("");
    requestAnimationFrame(() => questionInput.current?.focus());
  }

  const followUpPrompts = answer
    ? buildFollowUpPrompts(answer.citations.length ? answer.citations : answer.search_results)
    : [];

  return (
    <AppShell>
      <main className="page-container">
          {loadingDocument && <p className="mt-6 text-sm text-slate-500">Loading document...</p>}
          {documentError && <p role="alert" className="mt-6 rounded-lg bg-red-50 p-4 text-sm text-red-700">{documentError}</p>}
          {!loadingDocument && !documentError && document && (
            <>
              <PageHeader
                eyebrow="Documents / Detail"
                title={document.title || "Document"}
                description="Document analysis, metadata, and document-specific research."
                actions={<Link href="/documents" className="neutral-button">Back to Documents</Link>}
              />
              <div className="mt-6 grid gap-6 md:grid-cols-3">
                <section className="rounded-xl border border-slate-200 bg-white p-5 md:col-span-2">
                  <h2 className="font-medium text-slate-900">AI Analysis</h2>
                  {analysisError ? (
                    <p role="status" className="mt-3 text-sm text-slate-500">{analysisError}</p>
                  ) : analysis ? (
                    <>
                      <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-700">{analysis.summary || "No summary is available."}</p>
                      {!!analysis.important_sections.length && <p className="mt-4 text-xs text-slate-500">Sections: {analysis.important_sections.join(", ")}</p>}
                      {!!analysis.limitations.length && <p className="mt-3 text-xs text-amber-700">{analysis.limitations.join(" ")}</p>}
                    </>
                  ) : (
                    <p className="mt-3 text-sm text-slate-500">Analysis is not available for this document.</p>
                  )}
                </section>
                <aside className="rounded-xl border border-slate-200 bg-white p-5">
                  <h2 className="font-medium text-slate-900">Metadata</h2>
                  {[
                    ["Document type", document.document_type],
                    ["Court", document.court],
                    ["Case number", document.case_number],
                    ["Date", document.date],
                    ["Chunks", document.chunk_count],
                  ].map(([label, value]) => (
                    <p key={String(label)} className="mt-3 text-sm text-slate-600">
                      <span className="font-medium">{label}:</span> {value ?? "Not specified"}
                    </p>
                  ))}
                </aside>
              </div>
              <section className="mt-6 rounded-xl border border-slate-200 bg-white p-5">
                <h2 className="font-medium text-slate-900">Ask about this document</h2>
                <form onSubmit={ask} className="mt-4 flex gap-3">
                  <input
                    ref={questionInput}
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    disabled={asking}
                    maxLength={4000}
                    className="min-w-0 flex-1 rounded-lg border border-slate-300 px-4 py-3 text-sm"
                    placeholder="Ask a question restricted to this document"
                  />
                  <button disabled={asking || !query.trim()} className="rounded-lg bg-slate-900 px-5 text-sm font-medium text-white disabled:opacity-50">
                    {asking ? "Researching..." : "Ask"}
                  </button>
                </form>
                {questionError && <p role="alert" className="mt-3 text-sm text-red-700">{questionError}</p>}
                {asking && <p role="status" className="mt-4 text-sm text-slate-500">Searching this document...</p>}
                {answer && (
                  <div className="mt-5">
                    <p className="text-xs text-slate-500">Question: {askedQuery}</p>
                    <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-700">{answer.answer}</p>
                    {!answer.search_results.length && !answer.citations.length && (
                      <p className="mt-3 text-sm text-slate-500">No supporting passages were retrieved from this document.</p>
                    )}
                    {!!answer.citations.length && (
                      <div className="mt-5 space-y-3">
                        <h3 className="text-sm font-medium text-slate-800">Verified sources</h3>
                        {answer.citations.map((citation) => (
                          <article key={`${citation.document_id}:${citation.chunk_id}`} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                            <p className="text-sm font-medium text-slate-900">{citation.title || citation.document_id} · {citation.citation_id}</p>
                            {(citation.section || citation.subsection) && <p className="mt-1 text-xs text-teal-700">{[citation.section, citation.subsection].filter(Boolean).join(" · ")}</p>}
                            <p className="mt-1 text-xs text-slate-500">Chunk {citation.chunk_id}{citation.page_start != null ? ` · Page ${citation.page_start}${citation.page_end != null && citation.page_end !== citation.page_start ? `–${citation.page_end}` : ""}` : ""}</p>
                            {safeSourceUrl(citation.source_url) && <a className="mt-1 inline-block text-xs text-teal-700 hover:underline" href={safeSourceUrl(citation.source_url)!} target="_blank" rel="noopener noreferrer">View source</a>}
                          </article>
                        ))}
                      </div>
                    )}
                    {!!answer.limitations.length && <p className="mt-3 text-xs text-amber-700">{answer.limitations.join(" ")}</p>}
                    {followUpPrompts.length > 0 && (
                      <div className="follow-up-prompts mt-5" aria-label="Evidence-grounded follow-up questions">
                        <p className="follow-up-heading">Ask another question about this document</p>
                        {followUpPrompts.map((prompt) => (
                          <button
                            type="button"
                            key={`${prompt.documentId}:${prompt.label}`}
                            className="follow-up-button"
                            onClick={() => useFollowUp(prompt)}
                            disabled={asking}
                          >
                            {prompt.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </section>
            </>
          )}
      </main>
    </AppShell>
  );
}
