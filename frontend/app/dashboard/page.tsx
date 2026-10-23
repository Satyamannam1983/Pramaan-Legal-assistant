"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AppShell } from "../../components/AppShell";
import { PageHeader } from "../../components/PageHeader";
import { getErrorMessage, listDocuments, type DocumentSummary } from "../../lib/api";

function recentFirst(documents: DocumentSummary[]) {
  return [...documents].sort((left, right) => {
    const leftTime = left.created_at ? Date.parse(left.created_at) : 0;
    const rightTime = right.created_at ? Date.parse(right.created_at) : 0;
    return rightTime - leftTime;
  }).slice(0, 6);
}

function displayDate(value: string | null) {
  if (!value) return "Date unavailable";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "Date unavailable" : parsed.toLocaleDateString();
}

export default function DashboardPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    listDocuments(controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setDocuments(result);
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted) setError(getErrorMessage(failure));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  const recentDocuments = recentFirst(documents);

  return (
    <AppShell>
      <main className="page-container">
        <PageHeader
          eyebrow="Workspace overview"
          title="Dashboard"
          description="A clear view of your indexed legal documents and research tools."
          actions={<Link href="/" className="dark-button">Open Legal Assistant</Link>}
        />

        {error && <p className="assistant-error" role="alert">{error}</p>}

        <section aria-label="Indexed document count" className="dashboard-metric">
          <p className="dashboard-metric-label">Indexed documents</p>
          <p className="dashboard-metric-value" aria-live="polite">
            {loading ? "—" : error ? "Unavailable" : documents.length}
          </p>
          <p className="helper-text">Documents available to the current retrieval workspace</p>
        </section>

        <div className="dashboard-grid" style={{ marginTop: 30 }}>
          <section aria-labelledby="recent-documents-heading">
            <h2 id="recent-documents-heading" className="dashboard-section-title">Recent document activity</h2>
            <p className="helper-text" style={{ marginTop: 5 }}>Recent records from the indexed documents endpoint.</p>
            {loading ? (
              <p className="helper-text" role="status" style={{ marginTop: 18 }}>Loading indexed documents...</p>
            ) : !error && recentDocuments.length ? (
              <div className="document-list">
                {recentDocuments.map((doc) => (
                  <Link key={doc.document_id} href={`/documents/${encodeURIComponent(doc.document_id)}`} className="document-list-row">
                    <span>
                      <span className="document-list-title">{doc.title || doc.document_id}</span>
                      <span className="document-list-meta">
                        {doc.document_type || "Type unavailable"} · {doc.chunk_count} indexed chunks
                      </span>
                    </span>
                    <span className="document-list-date">{displayDate(doc.created_at)}</span>
                  </Link>
                ))}
              </div>
            ) : !error ? (
              <p className="helper-text" style={{ marginTop: 18 }}>No indexed documents are available yet. Upload a document to begin.</p>
            ) : null}
          </section>

          <section aria-labelledby="quick-access-heading">
            <h2 id="quick-access-heading" className="dashboard-section-title">Quick access</h2>
            <div className="shortcut-list">
              <Link href="/" className="shortcut-link"><span>Legal Assistant</span><span className="shortcut-arrow" aria-hidden="true">↗</span></Link>
              <Link href="/documents" className="shortcut-link"><span>Documents</span><span className="shortcut-arrow" aria-hidden="true">↗</span></Link>
              <Link href="/compare" className="shortcut-link"><span>Document Comparison</span><span className="shortcut-arrow" aria-hidden="true">↗</span></Link>
              <Link href="/evaluation" className="shortcut-link"><span>Evaluation</span><span className="shortcut-arrow" aria-hidden="true">↗</span></Link>
            </div>
            <p className="helper-text" style={{ marginTop: 16 }}>
              Recent queries and processing history are not provided by the current backend, so they are not shown here.
            </p>
          </section>
        </div>
      </main>
    </AppShell>
  );
}
