"use client";

import { useState } from "react";
import Link from "next/link";
import { AppShell } from "../components/AppShell";
import { API_URL as API_BASE } from "../lib/api";

const examples = [
  "What are the essential elements of a valid contract under Indian law?",
  "What are the remedies available for breach of contract?",
  "What is the difference between murder and culpable homicide?",
  "What does Section 302 IPC provide?",
];

type CitationRef = {
  citation_id: string;
  chunk_id: string;
  document_id: string;
  title: string | null;
  section: string | null;
  page_start: number | null;
  page_end: number | null;
  source_url: string | null;
  relevant_quote: string | null;
};

type ChatResponse = {
  answer: string;
  key_points: string[];
  citations: CitationRef[];
  limitations: string[];
  confidence: string;
  conversation_id: string;
  retrieval_attempts: number;
  strategies: string[];
};

type SearchResult = {
  chunk_id: string;
  content: string;
  document_id: string;
  title: string | null;
  section: string | null;
  page_start: number | null;
  page_end: number | null;
  document_type: string | null;
  court: string | null;
  date: string | null;
  source: string | null;
  source_url: string | null;
  vector_score: number;
  keyword_score: number;
  rrf_score: number;
  rerank_score: number;
  final_rank: number | null;
};

type SearchResponse = {
  query: string;
  results: SearchResult[];
};

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatResponse, setChatResponse] = useState<ChatResponse | null>(null);
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"answer" | "evidence">("answer");

  const handleSearch = async (q: string) => {
    const trimmed = q.trim();
    if (!trimmed) return;
    setQuery(trimmed);
    setLoading(true);
    setError(null);
    setChatResponse(null);
    setSearchResponse(null);

    try {
      // Run both chat and search in parallel
      const [chatRes, searchRes] = await Promise.allSettled([
        fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: trimmed }),
        }),
        fetch(`${API_BASE}/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: trimmed, top_k: 5 }),
        }),
      ]);

      // Handle chat response
      if (chatRes.status === "fulfilled" && chatRes.value.ok) {
        const data: ChatResponse = await chatRes.value.json();
        setChatResponse(data);
      }

      // Handle search response
      if (searchRes.status === "fulfilled" && searchRes.value.ok) {
        const data: SearchResponse = await searchRes.value.json();
        setSearchResponse(data);
      }

      // Check for errors
      if (chatRes.status === "fulfilled" && !chatRes.value.ok) {
        const detail = await chatRes.value.json().catch(() => ({ detail: chatRes.value.statusText }));
        setError(detail.detail || "Chat request failed");
      } else if (chatRes.status === "rejected") {
        setError("Could not connect to the API. Is the backend running?");
      }
    } catch (err) {
      setError("Could not connect to the API. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const confidenceColor = (c: string) => {
    if (c === "high") return "text-emerald-700 bg-emerald-50 border-emerald-200";
    if (c === "medium") return "text-amber-700 bg-amber-50 border-amber-200";
    return "text-slate-600 bg-slate-50 border-slate-200";
  };

  const rerankColor = (score: number) => {
    if (score >= 0.75) return "text-emerald-600 bg-emerald-50";
    if (score >= 0.50) return "text-amber-600 bg-amber-50";
    return "text-slate-500 bg-slate-100";
  };

  return (
    <AppShell eyebrow="Legal research">
    <main className="min-h-screen bg-slate-50">
      {/* Header */}
      <div className="mx-auto max-w-5xl px-6 py-10">
        {/* Search card */}
        <div className="rounded-2xl bg-slate-900 p-6 text-white shadow-sm">
          <p className="text-sm font-medium text-teal-400">Legal Research Assistant</p>
          <h1 className="mt-2 text-xl font-medium">
            Ask a legal question
          </h1>
          <p className="mt-1 text-xs text-slate-400">
            Hybrid retrieval + grounded LLM with verified citations
          </p>
          <div className="mt-5 flex gap-3">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch(query)}
              placeholder="What are the essential requirements for a valid contract?"
              className="min-w-0 flex-1 rounded-lg border border-white/20 bg-white/10 px-4 py-3 text-sm outline-none placeholder:text-slate-400 focus:border-amber-400"
            />
            <button
              onClick={() => handleSearch(query)}
              disabled={loading}
              className="rounded-lg bg-amber-400 px-5 py-3 text-sm font-semibold text-slate-900 disabled:opacity-50"
            >
              {loading ? "Researching..." : "Ask"}
            </button>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            {examples.map((ex) => (
              <button
                key={ex}
                onClick={() => handleSearch(ex)}
                className="rounded-full border border-white/20 px-3 py-1 text-xs text-slate-300 hover:border-amber-400 hover:text-amber-400"
              >
                {ex.length > 55 ? ex.slice(0, 52) + "..." : ex}
              </button>
            ))}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="mt-8 text-center text-sm text-slate-500">
            Retrieving evidence and generating answer...
          </div>
        )}

        {/* Results */}
        {chatResponse && !loading && (
          <section className="mt-8 space-y-6">
            {/* Tab bar */}
            <div className="flex gap-1 border-b border-slate-200">
              <button
                onClick={() => setActiveTab("answer")}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === "answer"
                    ? "border-teal-500 text-teal-600"
                    : "border-transparent text-slate-500 hover:text-slate-700"
                }`}
              >
                AI Answer
              </button>
              <button
                onClick={() => setActiveTab("evidence")}
                className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === "evidence"
                    ? "border-teal-500 text-teal-600"
                    : "border-transparent text-slate-500 hover:text-slate-700"
                }`}
              >
                Evidence ({searchResponse?.results.length || 0})
              </button>
            </div>

            {/* AI Answer tab */}
            {activeTab === "answer" && (
              <div className="space-y-5">
                {/* Confidence badge */}
                <div className="flex items-center gap-3">
                  <span className={`rounded-full border px-3 py-1 text-xs font-medium ${confidenceColor(chatResponse.confidence)}`}>
                    Confidence: {chatResponse.confidence}
                  </span>
                  <span className="text-xs text-slate-400">
                    {chatResponse.citations.length} source{chatResponse.citations.length !== 1 ? "s" : ""} cited
                  </span>
                  {chatResponse.retrieval_attempts > 0 && (
                    <span className="text-xs text-slate-400">
                      {chatResponse.retrieval_attempts} retrieval pass{chatResponse.retrieval_attempts !== 1 ? "es" : ""}
                    </span>
                  )}
                </div>

                {/* Answer */}
                <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                  <h2 className="text-sm font-medium text-slate-500 mb-3">Answer</h2>
                  <div className="prose prose-sm max-w-none text-slate-700 leading-relaxed whitespace-pre-wrap">
                    {chatResponse.answer}
                  </div>
                </div>

                {/* Key Points */}
                {chatResponse.key_points.length > 0 && (
                  <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                    <h2 className="text-sm font-medium text-slate-500 mb-3">Key Points</h2>
                    <ul className="space-y-2">
                      {chatResponse.key_points.map((point, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                          <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-teal-500" />
                          {point}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Citations */}
                {chatResponse.citations.length > 0 && (
                  <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                    <h2 className="text-sm font-medium text-slate-500 mb-3">Sources Cited</h2>
                    <div className="space-y-3">
                      {chatResponse.citations.map((cit, i) => (
                        <div key={i} className="rounded-lg border border-slate-100 bg-slate-50 p-4">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="rounded bg-teal-100 px-2 py-0.5 text-xs font-bold text-teal-700">
                              {cit.citation_id}
                            </span>
                            <span className="font-medium text-sm text-slate-900">
                              {cit.title || cit.document_id}
                            </span>
                          </div>
                          {cit.section && (
                            <p className="text-xs text-teal-600 mb-1">{cit.section}</p>
                          )}
                          <div className="flex flex-wrap gap-2 text-xs text-slate-400">
                            {cit.page_start != null && (
                              <span>Page {cit.page_start === cit.page_end ? cit.page_start : `${cit.page_start}–${cit.page_end}`}</span>
                            )}
                            {cit.source_url && (
                              <a href={cit.source_url} target="_blank" rel="noopener noreferrer" className="text-teal-600 hover:underline">
                                View source
                              </a>
                            )}
                          </div>
                          {cit.relevant_quote && (
                            <blockquote className="mt-2 border-l-2 border-teal-300 pl-3 text-xs text-slate-600 italic">
                              &ldquo;{cit.relevant_quote}&rdquo;
                            </blockquote>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Limitations */}
                {chatResponse.limitations.length > 0 && (
                  <div className="rounded-xl border border-amber-200 bg-amber-50 p-5">
                    <h2 className="text-sm font-medium text-amber-700 mb-2">Limitations</h2>
                    <ul className="space-y-1">
                      {chatResponse.limitations.map((lim, i) => (
                        <li key={i} className="text-xs text-amber-600">{lim}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {/* Evidence tab */}
            {activeTab === "evidence" && searchResponse && (
              <div className="space-y-4">
                {searchResponse.results.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center">
                    <p className="text-slate-500">No evidence retrieved.</p>
                  </div>
                ) : (
                  searchResponse.results.map((result, idx) => (
                    <article
                      key={result.chunk_id}
                      className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-xs font-medium text-slate-400">
                              #{result.final_rank || idx + 1}
                            </span>
                            {result.document_type && (
                              <span className="rounded bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
                                {result.document_type}
                              </span>
                            )}
                            {result.court && (
                              <span className="rounded bg-blue-50 px-2 py-0.5 text-xs text-blue-700">
                                {result.court}
                              </span>
                            )}
                          </div>
                          <h3 className="mt-1 font-semibold text-slate-900">
                            {result.title || result.document_id}
                          </h3>
                          {result.section && (
                            <p className="text-sm text-teal-700 font-medium mt-0.5">
                              {result.section}
                            </p>
                          )}
                        </div>
                        <div className={`flex-shrink-0 rounded-lg px-3 py-1.5 text-center ${rerankColor(result.rerank_score)}`}>
                          <div className="text-xs font-medium">Relevance</div>
                          <div className="text-sm font-bold">
                            {result.rerank_score > 0 && result.rerank_score <= 1
                              ? (result.rerank_score * 100).toFixed(0) + "%"
                              : result.rerank_score > 1
                                ? result.rerank_score.toFixed(2)
                                : "N/A"}
                          </div>
                        </div>
                      </div>

                      <blockquote className="mt-4 rounded-lg bg-slate-50 border-l-4 border-teal-400 px-4 py-3 text-sm text-slate-700 leading-relaxed">
                        {result.content.length > 600
                          ? result.content.slice(0, 597) + "..."
                          : result.content}
                      </blockquote>

                      <details className="mt-3 group">
                        <summary className="cursor-pointer text-xs text-slate-400 hover:text-slate-600 select-none">
                          Why this result?
                        </summary>
                        <div className="mt-2 flex flex-wrap gap-3 text-xs">
                          <span className="rounded bg-violet-50 px-2 py-1 text-violet-700">
                            Vector: {(result.vector_score * 100).toFixed(1)}%
                          </span>
                          <span className="rounded bg-orange-50 px-2 py-1 text-orange-700">
                            Keyword: {result.keyword_score.toFixed(2)}
                          </span>
                          <span className="rounded bg-cyan-50 px-2 py-1 text-cyan-700">
                            RRF: {result.rrf_score.toFixed(4)}
                          </span>
                          <span className="rounded bg-emerald-50 px-2 py-1 text-emerald-700">
                            Reranker: {result.rerank_score.toFixed(4)}
                          </span>
                        </div>
                      </details>

                      <footer className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400">
                        {result.page_start != null && (
                          <span>
                            Page {result.page_start === result.page_end
                              ? result.page_start
                              : `${result.page_start}–${result.page_end}`}
                          </span>
                        )}
                        {result.date && <span>Date: {result.date}</span>}
                        {result.source && (
                          <span>
                            Source:{" "}
                            {result.source_url ? (
                              <a href={result.source_url} target="_blank" rel="noopener noreferrer" className="text-teal-600 hover:underline">
                                {result.source}
                              </a>
                            ) : (
                              result.source
                            )}
                          </span>
                        )}
                        <span className="ml-auto font-mono text-slate-300">
                          {result.chunk_id}
                        </span>
                      </footer>
                    </article>
                  ))
                )}
              </div>
            )}
          </section>
        )}

        {/* Empty state */}
        {!chatResponse && !loading && !error && (
          <div className="mt-10 rounded-xl border border-dashed border-slate-200 bg-white p-10 text-center">
            <p className="text-slate-500">Ask a legal question to get a grounded answer with citations.</p>
            <p className="mt-2 text-xs text-slate-400">
              The system retrieves evidence from indexed legal documents, then generates an answer grounded in those sources.
            </p>
          </div>
        )}
      </div>
    </main>
    </AppShell>
  );
}
