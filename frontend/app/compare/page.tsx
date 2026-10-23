"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { AppShell } from "../../components/AppShell";
import { PageHeader } from "../../components/PageHeader";
import {
  compareDocuments,
  getErrorMessage,
  listDocuments,
  type DocumentComparisonResponse,
  type DocumentSummary,
} from "../../lib/api";

export default function ComparePage() {
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [loadingDocuments, setLoadingDocuments] = useState(true);
  const [documentsError, setDocumentsError] = useState("");
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [result, setResult] = useState<DocumentComparisonResponse | null>(null);
  const [comparisonError, setComparisonError] = useState("");
  const [comparing, setComparing] = useState(false);
  const comparisonController = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    listDocuments(controller.signal)
      .then(setDocs)
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setDocumentsError(getErrorMessage(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingDocuments(false);
      });
    return () => {
      controller.abort();
      const activeComparison = comparisonController.current;
      comparisonController.current = null;
      activeComparison?.abort();
    };
  }, []);

  async function compare() {
    if (!a || !b || a === b || comparing) return;
    setComparing(true);
    setComparisonError("");
    setResult(null);
    const controller = new AbortController();
    comparisonController.current = controller;
    try {
      const comparison = await compareDocuments(
        { document_ids: [a, b] },
        controller.signal,
      );
      if (comparisonController.current === controller) setResult(comparison);
    } catch (error) {
      if (!controller.signal.aborted && comparisonController.current === controller) {
        setComparisonError(getErrorMessage(error));
      }
    } finally {
      if (comparisonController.current === controller) {
        comparisonController.current = null;
        setComparing(false);
      }
    }
  }

  const select = (value: string, set: (v: string) => void, label: string) => (
    <select
      aria-label={label}
      value={value}
      onChange={(event) => {
        set(event.target.value);
        setResult(null);
        setComparisonError("");
      }}
      disabled={loadingDocuments || comparing}
      className="w-full rounded-lg border border-slate-300 px-3 py-3 text-sm"
    >
      <option value="">Select document</option>
      {docs.map((doc) => <option key={doc.document_id} value={doc.document_id}>{doc.title || doc.document_id}</option>)}
    </select>
  );

  return (
    <AppShell>
      <main className="page-container">
          <PageHeader
            eyebrow="Workspace / Comparison"
            title="Document Comparison"
            description="Select two indexed documents to inspect their similarities, differences, provisions, and source passages."
            actions={<Link href="/documents" className="neutral-button">Browse documents</Link>}
          />
          <div className="mt-8 grid gap-4 rounded-xl border border-slate-200 bg-white p-5 md:grid-cols-2">
            {select(a, setA, "First document")}
            {select(b, setB, "Second document")}
            <button
              disabled={!a || !b || a === b || comparing || loadingDocuments}
              onClick={compare}
              className="rounded-lg bg-slate-900 px-5 py-3 text-sm font-medium text-white disabled:opacity-40 md:col-span-2"
            >
              {comparing ? "Comparing..." : "Compare"}
            </button>
          </div>
          {loadingDocuments && <p role="status" className="mt-4 text-sm text-slate-500">Loading documents...</p>}
          {documentsError && <p role="alert" className="mt-4 rounded-lg bg-red-50 p-4 text-sm text-red-700">{documentsError}</p>}
          {!loadingDocuments && !documentsError && docs.length < 2 && (
            <p className="mt-4 text-sm text-slate-500">At least two indexed documents are required to compare.</p>
          )}
          {comparisonError && <p role="alert" className="mt-4 rounded-lg bg-red-50 p-4 text-sm text-red-700">{comparisonError}</p>}
          {result && (
            <section className="mt-6 space-y-5 rounded-xl border border-slate-200 bg-white p-6">
              <div>
                <h2 className="font-medium">Overview</h2>
                <p className="mt-2 text-sm text-slate-700">{result.overview}</p>
              </div>
              {[
                ["Similarities", result.similarities],
                ["Differences", result.differences],
                ["Legal provisions", result.legal_provisions],
                ["Reasoning comparison", result.reasoning_comparison],
              ].map(([heading, items]) => (
                <div key={String(heading)}>
                  <h2 className="font-medium">{heading}</h2>
                  {Array.isArray(items) && items.length ? (
                    <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-700">
                      {items.map((item, index) => <li key={`${heading}-${index}`}>{item}</li>)}
                    </ul>
                  ) : <p className="mt-2 text-sm text-slate-500">No comparison findings were returned.</p>}
                </div>
              ))}
              {result.outcome_comparison && (
                <div><h2 className="font-medium">Outcome</h2><p className="mt-2 text-sm text-slate-700">{result.outcome_comparison}</p></div>
              )}
              {!!result.citations.length && (
                <div>
                  <h2 className="font-medium">Source passages</h2>
                  <div className="mt-2 space-y-3">
                    {result.citations.map((citation) => (
                      <article key={`${citation.citation_id}:${citation.chunk_id}`} className="rounded-lg bg-slate-50 p-3">
                        <p className="text-xs font-medium text-teal-700">{citation.citation_id} · {citation.document_id}</p>
                        {(citation.section || citation.page_start != null) && (
                          <p className="mt-1 text-xs text-slate-500">
                            {citation.section || "Section unavailable"}
                            {citation.page_start != null ? ` · Page ${citation.page_start}${citation.page_end != null && citation.page_end !== citation.page_start ? `–${citation.page_end}` : ""}` : ""}
                          </p>
                        )}
                        {citation.relevant_quote && <blockquote className="mt-2 text-sm text-slate-700">{citation.relevant_quote}</blockquote>}
                      </article>
                    ))}
                  </div>
                </div>
              )}
              {!!result.limitations.length && <p className="text-xs text-amber-700">{result.limitations.join(" ")}</p>}
            </section>
          )}
      </main>
    </AppShell>
  );
}
