"use client";

import { useEffect, useState } from "react";
import { AppShell } from "../../components/AppShell";

import { API_URL } from "../../lib/api";

type Report = {
  status: string;
  dataset_size: number;
  metrics: Record<string, Record<string, number>>;
  limitations: string[];
};

export default function EvaluationPage() {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    fetch(`${API_URL}/evaluation`)
      .then((response) => response.ok ? response.json() : Promise.reject())
      .then(setReport)
      .catch(() => setError(true));
  }, []);
  const unavailable = error || !report || report.status === "NOT RUN";
  return <AppShell eyebrow="Development / evaluation">
    <main className="px-6 py-10">
      <div className="mx-auto max-w-5xl">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-semibold">Legal RAG Evaluation</h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
              Retrieval, citation, and grounding metrics for development only. These metrics do not measure legal correctness.
            </p>
          </div>
          {report && <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600">{report.status}</span>}
        </div>
        {unavailable ? (
          <section className="mt-8 rounded-xl border border-amber-200 bg-amber-50 p-6">
            <h2 className="font-medium text-amber-950">Evaluation not available</h2>
            <p className="mt-2 text-sm leading-6 text-amber-900">
              MongoDB or the required model infrastructure is unavailable. Run the evaluation after those services are ready; unavailable benchmarks are not shown as zero scores.
            </p>
          </section>
        ) : (
          <>
            <div className="mt-8 grid gap-5 sm:grid-cols-3">
              <MetricCard label="Dataset size" value={String(report.dataset_size)} />
              <MetricCard label="Retrieval" value={report.metrics.retrieval ? "Measured" : "Not run"} />
              <MetricCard label="Grounding" value={report.metrics.rag ? "Measured" : "Not run"} />
            </div>
            <section className="mt-6 rounded-xl border border-slate-200 bg-white p-6">
              <h2 className="font-medium">Metrics</h2>
              {Object.entries(report.metrics).map(([name, values]) => (
                <div key={name} className="mt-5">
                  <h3 className="text-sm font-medium capitalize text-slate-700">{name}</h3>
                  <div className="mt-2 grid gap-2 sm:grid-cols-3">
                    {Object.entries(values).map(([key, value]) => (
                      <div key={key} className="rounded-lg bg-slate-50 p-3 text-xs">
                        <span className="text-slate-500">{key.replaceAll("_", " ")}</span>
                        <strong className="ml-2">{value.toFixed(3)}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </section>
          </>
        )}
        {!!report?.limitations?.length && <section className="mt-6 rounded-xl border border-slate-200 bg-white p-5 text-sm text-slate-600"><h2 className="font-medium text-slate-900">Limitations</h2>{report.limitations.map((item) => <p key={item} className="mt-2">{item}</p>)}</section>}
      </div>
    </main>
  </AppShell>;
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl border border-slate-200 bg-white p-5"><p className="text-xs text-slate-500">{label}</p><p className="mt-2 text-2xl font-semibold">{value}</p></div>;
}
