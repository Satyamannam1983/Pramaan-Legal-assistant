"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { AppShell } from "../../../components/AppShell";
import { API_URL as API } from "../../../lib/api";

export default function DocumentDetailPage() {
  const params = useParams<{ document_id: string }>();
  const [document, setDocument] = useState<Record<string, unknown> | null>(null);
  const [analysis, setAnalysis] = useState<Record<string, unknown> | null>(null);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  useEffect(() => {
    if (!params.document_id) return;
    Promise.all([
      fetch(`${API}/documents/${params.document_id}`).then((r) => r.json()),
      fetch(`${API}/documents/${params.document_id}/analysis`).then((r) => r.ok ? r.json() : null),
    ]).then(([doc, result]) => { setDocument(doc); setAnalysis(result); });
  }, [params.document_id]);

  async function ask() {
    const response = await fetch(`${API}/documents/${params.document_id}/ask`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const result = await response.json();
    setAnswer(result.answer || result.detail || "No answer available.");
  }

  return (
    <AppShell eyebrow="Document workspace"><main className="px-6 py-10">
      <div className="mx-auto max-w-5xl">
        <Link href="/documents" className="text-sm text-teal-700">← Documents</Link>
        <h1 className="mt-6 text-3xl font-semibold text-slate-900">{String(document?.title || "Document")}</h1>
        <div className="mt-6 grid gap-6 md:grid-cols-3">
          <section className="rounded-xl border border-slate-200 bg-white p-5 md:col-span-2">
            <h2 className="font-medium text-slate-900">AI Analysis</h2>
            <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-700">{String(analysis?.summary || "Analysis is not available until this document is indexed.")}</p>
            {!!analysis?.important_sections && <p className="mt-4 text-xs text-slate-500">Sections: {(analysis.important_sections as string[]).join(", ")}</p>}
          </section>
          <aside className="rounded-xl border border-slate-200 bg-white p-5">
            <h2 className="font-medium text-slate-900">Metadata</h2>
            {["document_type", "court", "case_number", "date", "chunk_count"].map((key) => (
              <p key={key} className="mt-3 text-sm text-slate-600"><span className="font-medium">{key.replace("_", " ")}:</span> {String(document?.[key] || "Not specified")}</p>
            ))}
          </aside>
        </div>
        <section className="mt-6 rounded-xl border border-slate-200 bg-white p-5">
          <h2 className="font-medium text-slate-900">Ask about this document</h2>
          <div className="mt-4 flex gap-3">
            <input value={query} onChange={(e) => setQuery(e.target.value)} className="min-w-0 flex-1 rounded-lg border border-slate-300 px-4 py-3 text-sm" placeholder="Ask a question restricted to this document" />
            <button onClick={ask} className="rounded-lg bg-slate-900 px-5 text-sm font-medium text-white">Ask</button>
          </div>
          {answer && <p className="mt-5 whitespace-pre-wrap text-sm leading-6 text-slate-700">{answer}</p>}
        </section>
      </div>
    </main></AppShell>
  );
}
