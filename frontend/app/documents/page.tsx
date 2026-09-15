"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AppShell } from "../../components/AppShell";

type DocumentSummary = {
  document_id: string;
  title: string | null;
  document_type: string | null;
  court: string | null;
  date: string | null;
  chunk_count: number;
  created_at: string | null;
};

import { API_URL as API } from "../../lib/api";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    fetch(`${API}/documents`)
      .then((response) => response.ok ? response.json() : Promise.reject())
      .then(setDocuments)
      .catch(() => setError("The document store is unavailable."));
  }, []);

  return (
    <AppShell eyebrow="Legal documents">
    <main className="px-6 py-10">
      <div className="mx-auto max-w-5xl">
        <Link href="/" className="text-sm text-teal-700">← Research</Link>
        <h1 className="mt-6 text-3xl font-semibold text-slate-900">Legal Documents</h1>
        <p className="mt-2 text-sm text-slate-500">Uploaded and indexed documents available for grounded analysis.</p>
        {error && <p className="mt-6 rounded-lg bg-red-50 p-4 text-sm text-red-700">{error}</p>}
        <div className="mt-8 overflow-hidden rounded-xl border border-slate-200 bg-white">
          {documents.map((doc) => (
            <Link key={doc.document_id} href={`/documents/${doc.document_id}`} className="block border-b border-slate-100 p-5 hover:bg-slate-50">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h2 className="font-medium text-slate-900">{doc.title || doc.document_id}</h2>
                  <p className="mt-1 text-xs text-slate-500">{doc.document_type || "Unknown"} · {doc.court || "Court not specified"} · {doc.chunk_count} chunks</p>
                </div>
                <span className="text-xs text-slate-400">{doc.date || "Date unavailable"}</span>
              </div>
            </Link>
          ))}
          {!documents.length && !error && <p className="p-6 text-sm text-slate-500">No indexed documents yet.</p>}
        </div>
      </div>
    </main>
    </AppShell>
  );
}
