"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { AppShell } from "../../components/AppShell";
import { API_URL as API } from "../../lib/api";

type Doc = { document_id: string; title: string | null; document_type: string | null };

export default function ComparePage() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  useEffect(() => { fetch(`${API}/documents`).then((r) => r.json()).then(setDocs).catch(() => setDocs([])); }, []);
  async function compare() {
    const response = await fetch(`${API}/documents/compare`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document_ids: [a, b] }),
    });
    setResult(await response.json());
  }
  const select = (value: string, set: (v: string) => void) => (
    <select value={value} onChange={(e) => set(e.target.value)} className="w-full rounded-lg border border-slate-300 px-3 py-3 text-sm">
      <option value="">Select document</option>
      {docs.map((doc) => <option key={doc.document_id} value={doc.document_id}>{doc.title || doc.document_id}</option>)}
    </select>
  );
  return (
    <AppShell eyebrow="Document comparison"><main className="px-6 py-10"><div className="mx-auto max-w-5xl">
      <Link href="/documents" className="text-sm text-teal-700">← Documents</Link>
      <h1 className="mt-6 text-3xl font-semibold text-slate-900">Compare Legal Documents</h1>
      <div className="mt-8 grid gap-4 rounded-xl border border-slate-200 bg-white p-5 md:grid-cols-2">{select(a, setA)}{select(b, setB)}
        <button disabled={!a || !b || a === b} onClick={compare} className="rounded-lg bg-slate-900 px-5 py-3 text-sm font-medium text-white disabled:opacity-40 md:col-span-2">Compare</button>
      </div>
      {result && <section className="mt-6 space-y-4 rounded-xl border border-slate-200 bg-white p-6">
        <h2 className="font-medium">Overview</h2><p className="text-sm text-slate-700">{String(result.overview || result.detail || "")}</p>
        <h2 className="font-medium">Similarities</h2>{(result.similarities as string[] || []).map((item) => <p key={item} className="text-sm text-slate-700">• {item}</p>)}
        <p className="text-xs text-slate-500">{(result.limitations as string[] || []).join(" ")}</p>
      </section>}
    </div></main></AppShell>
  );
}
