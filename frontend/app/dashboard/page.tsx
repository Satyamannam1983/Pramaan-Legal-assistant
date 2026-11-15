"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AppShell } from "../../components/AppShell";
import { PageHeader } from "../../components/PageHeader";
import {
  getCorpusStats,
  getErrorMessage,
  listDocuments,
  type CorpusStats,
  type DocumentSummary,
} from "../../lib/api";

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

function formatYearCoverage(years: number[]) {
  const sorted = Array.from(new Set(years)).sort((left, right) => left - right);
  if (!sorted.length) return "not recorded";
  const ranges: Array<[number, number]> = [];
  for (const year of sorted) {
    const lastRange = ranges[ranges.length - 1];
    if (lastRange && year === lastRange[1] + 1) lastRange[1] = year;
    else ranges.push([year, year]);
  }
  return ranges.map(([start, end]) => start === end ? `${start}` : `${start}-${end}`).join(", ");
}

export default function DashboardPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [corpusStats, setCorpusStats] = useState<CorpusStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    const refreshStats = () => {
      getCorpusStats(controller.signal)
        .then((result) => {
          if (!controller.signal.aborted) {
            setCorpusStats(result);
            setStatsError("");
          }
        })
        .catch((failure: unknown) => {
          if (!controller.signal.aborted) setStatsError(getErrorMessage(failure));
        })
        .finally(() => {
          if (!controller.signal.aborted) setStatsLoading(false);
        });
    };
    refreshStats();
    const statsInterval = window.setInterval(refreshStats, 10_000);
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
    return () => {
      window.clearInterval(statsInterval);
      controller.abort();
    };
  }, []);

  const recentDocuments = recentFirst(documents);
  const metricValue = (value: number | null | undefined) => {
    if (statsLoading) return "—";
    if (statsError || value === undefined) return "Unavailable";
    return value === null ? "Not counted" : value.toLocaleString();
  };

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

        <section aria-labelledby="corpus-stats-heading" className="corpus-overview">
          <div className="corpus-overview-heading">
            <div>
              <h2 id="corpus-stats-heading" className="dashboard-section-title">Legal corpus</h2>
              <p className="helper-text">Live totals from the existing document index and local ingestion status.</p>
            </div>
            {corpusStats && (
              <span className="corpus-status" role="status">
                Ingestion: {corpusStats.ingestion.status.replace(/_/g, " ")}
              </span>
            )}
          </div>
          {statsError && <p className="assistant-error" role="alert">{statsError}</p>}
          <div className="corpus-metrics" aria-live="polite">
            <article className="corpus-metric-card">
              <p className="corpus-metric-label">Indexed documents</p>
              <p className="corpus-metric-value">{metricValue(corpusStats?.indexed_documents)}</p>
            </article>
            <article className="corpus-metric-card">
              <p className="corpus-metric-label">Indexed chunks</p>
              <p className="corpus-metric-value">{metricValue(corpusStats?.indexed_chunks)}</p>
            </article>
            <article className="corpus-metric-card">
              <p className="corpus-metric-label">Available metadata records</p>
              <p className="corpus-metric-value">{metricValue(corpusStats?.metadata_records_available)}</p>
            </article>
            <article className="corpus-metric-card">
              <p className="corpus-metric-label">Downloaded archives</p>
              <p className="corpus-metric-value">{metricValue(corpusStats?.archives_downloaded)}</p>
            </article>
            <article className="corpus-metric-card">
              <p className="corpus-metric-label">Extracted PDFs</p>
              <p className="corpus-metric-value">{metricValue(corpusStats?.pdfs_extracted)}</p>
            </article>
          </div>
          <div className="corpus-breakdown">
            <div>
              <strong>Supreme Court</strong>
              <span>{metricValue(corpusStats?.indexed_by_court.supreme_court.documents)} documents · {metricValue(corpusStats?.indexed_by_court.supreme_court.chunks)} chunks</span>
            </div>
            <div>
              <strong>High Courts</strong>
              <span>{metricValue(corpusStats?.indexed_by_court.high_courts.documents)} documents · {metricValue(corpusStats?.indexed_by_court.high_courts.chunks)} chunks</span>
            </div>
          </div>
          {corpusStats && (
            <p className="corpus-progress">
              Coverage: Supreme Court English {formatYearCoverage(corpusStats.archive_coverage.supreme_court_english_years)}
              {" · "}High Court archives: {corpusStats.archive_coverage.high_court_year_courts.length
                ? corpusStats.archive_coverage.high_court_year_courts.map((archive) => `${archive.year} (court ${archive.court_id})`).join(", ")
                : "not recorded"}
            </p>
          )}
          {corpusStats && (
            <p className="corpus-progress">
              Processing {corpusStats.ingestion.documents_processed.toLocaleString()} of {corpusStats.ingestion.documents_to_index.toLocaleString()}
              {" · "}{corpusStats.ingestion.failed_documents.toLocaleString()} failed
              {" · "}{corpusStats.ingestion.duplicates_skipped.toLocaleString()} duplicates skipped
              {" · "}{corpusStats.ingestion.pending_documents.toLocaleString()} pending
              {" · "}metadata: {metricValue(corpusStats.metadata_records_by_dataset.supreme_court)} Supreme Court, {metricValue(corpusStats.metadata_records_by_dataset.high_courts)} High Courts
            </p>
          )}
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
