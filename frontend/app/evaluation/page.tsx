"use client";

import { useEffect, useState } from "react";
import { AppShell } from "../../components/AppShell";
import { PageHeader } from "../../components/PageHeader";
import {
  getErrorMessage,
  getEvaluationReport,
  type EvaluationReport,
  type EvaluationStage,
} from "../../lib/api";

export default function EvaluationPage() {
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    getEvaluationReport(controller.signal)
      .then(setReport)
      .catch((failure: unknown) => {
        if (!controller.signal.aborted) setError(getErrorMessage(failure));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  const evaluations = Object.entries(report?.evaluations ?? {});

  return (
    <AppShell>
      <main className="page-container">
          <PageHeader
            eyebrow="Analytics"
            title="Evaluation"
            description="Retrieval rankings and operational observations are reported separately from answer quality. Citation checks validate identifiers and provenance, not whether a source entails a legal claim."
            actions={report && <StatusBadge status={report.status} />}
          />
          {loading && <p role="status" className="mt-8 text-sm text-slate-500">Loading evaluation report...</p>}
          {error && <p role="alert" className="mt-8 rounded-lg bg-red-50 p-4 text-sm text-red-700">{error}</p>}

          {report && (
            <>
              <div className="mt-8 grid gap-5 sm:grid-cols-3">
                <MetricCard label="Dataset examples" value={String(report.dataset_size)} />
                <MetricCard label="Evaluation stages" value={String(evaluations.length)} />
                <MetricCard label="Generated at" value={formatTimestamp(report.timestamp)} />
              </div>

              {report.status === "NOT RUN" && (
                <section className="mt-6 rounded-xl border border-amber-200 bg-amber-50 p-6">
                  <h2 className="font-medium text-amber-950">Evaluation has not been run</h2>
                  <p className="mt-2 text-sm leading-6 text-amber-900">
                    No benchmark results are available. Missing metrics are not represented as zero.
                  </p>
                </section>
              )}

              {!!Object.keys(report.datasets ?? {}).length && (
                <section className="mt-8">
                  <h2 className="text-lg font-semibold">Datasets</h2>
                  <div className="mt-3 divide-y divide-slate-200 rounded-xl border border-slate-200 bg-white">
                    {Object.entries(report.datasets ?? {}).map(([name, dataset]) => (
                      <div key={name} className="p-4">
                        <div className="flex flex-wrap items-baseline justify-between gap-2">
                          <h3 className="font-medium">{dataset.name} <span className="text-xs text-slate-500">v{dataset.version}</span></h3>
                          <span className="text-xs text-slate-500">{dataset.example_count} examples</span>
                        </div>
                        <p className="mt-2 text-sm text-slate-600">{dataset.provenance}</p>
                        <p className="mt-2 break-all font-mono text-xs text-slate-500">SHA-256: {dataset.sha256}</p>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {!!report.configuration && (
                <section className="mt-8">
                  <h2 className="text-lg font-semibold">Run configuration</h2>
                  <div className="mt-3 grid gap-2 rounded-xl border border-slate-200 bg-white p-4 sm:grid-cols-2">
                    {Object.entries(report.configuration).map(([key, value]) => (
                      <div key={key} className="flex justify-between gap-4 text-xs">
                        <span className="text-slate-500">{key.replaceAll("_", " ")}</span>
                        <span className="text-right font-medium">{String(value)}</span>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {!!evaluations.length && (
                <section className="mt-8">
                  <h2 className="text-lg font-semibold">Evaluation stages</h2>
                  <div className="mt-3 space-y-4">
                    {evaluations.map(([name, stage]) => (
                      <EvaluationStageView key={name} name={name} stage={stage} />
                    ))}
                  </div>
                </section>
              )}

              {!!report.limitations?.length && (
                <section className="mt-8 rounded-xl border border-slate-200 bg-white p-5 text-sm text-slate-600">
                  <h2 className="font-medium text-slate-900">Limitations</h2>
                  {report.limitations.map((item, index) => <p key={`${index}-${item}`} className="mt-2">{item}</p>)}
                </section>
              )}
            </>
          )}
      </main>
    </AppShell>
  );
}

function EvaluationStageView({ name, stage }: { name: string; stage: EvaluationStage }) {
  const metricEntries = [
    ...flatten(stage.metrics ?? {}),
    ...flattenCategories(stage.category_metrics ?? {}),
    ...flatten(stage.operational_metrics ?? {}),
  ];
  const diagnosticEntries = flatten(stage.diagnostics ?? {});

  return (
    <article className="rounded-xl border border-slate-200 bg-white p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-medium">{name.replaceAll("_", " ")}</h3>
        <StatusBadge status={stage.status} />
      </div>
      {stage.reason && <p className="mt-3 text-sm leading-6 text-slate-600">{stage.reason}</p>}
      {metricEntries.length > 0 && (
        <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {metricEntries.map(([key, value]) => (
            <div key={key} className="rounded-lg bg-slate-50 p-3 text-xs">
              <span className="block text-slate-500">{key.replaceAll("_", " ")}</span>
              <strong className="mt-1 block">{formatValue(value)}</strong>
            </div>
          ))}
        </div>
      )}
      {diagnosticEntries.length > 0 && (
        <details className="mt-4">
          <summary className="cursor-pointer text-sm text-slate-600">Execution diagnostics</summary>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {diagnosticEntries.map(([key, value]) => (
              <div key={key} className="flex justify-between gap-4 text-xs">
                <span className="text-slate-500">{key.replaceAll("_", " ")}</span>
                <span className="text-right">{formatValue(value)}</span>
              </div>
            ))}
          </div>
        </details>
      )}
      {stage.quality_evaluation && (
        <div className="mt-4 border-t border-slate-200 pt-4">
          <div className="flex items-center justify-between gap-3">
            <span className="text-sm font-medium">Semantic answer quality</span>
            <StatusBadge status={stage.quality_evaluation.status} />
          </div>
          {stage.quality_evaluation.reason && (
            <p className="mt-2 text-sm text-slate-600">{stage.quality_evaluation.reason}</p>
          )}
        </div>
      )}
    </article>
  );
}

function flatten(values: Record<string, unknown>): Array<[string, unknown]> {
  const entries: Array<[string, unknown]> = [];
  for (const [key, value] of Object.entries(values)) {
    if (value === null || value === undefined) continue;
    if (typeof value === "object" && !Array.isArray(value)) {
      for (const [child, childValue] of flatten(value as Record<string, unknown>)) {
        entries.push([`${key}_${child}`, childValue]);
      }
    } else if (Array.isArray(value)) {
      entries.push([key, JSON.stringify(value)]);
    } else {
      entries.push([key, value]);
    }
  }
  return entries;
}

function flattenCategories(
  categories: Record<string, Record<string, unknown>>,
): Array<[string, unknown]> {
  const entries: Array<[string, unknown]> = [];
  for (const [category, metrics] of Object.entries(categories)) {
    for (const [name, value] of flatten(metrics)) {
      entries.push([`${category}_${name}`, value]);
    }
  }
  return entries;
}

function formatValue(value: unknown): string {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  return String(value);
}

function formatTimestamp(value?: string): string {
  if (!value) return "Not recorded";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function StatusBadge({ status }: { status: string }) {
  const style = status === "COMPLETED"
    ? "border-emerald-200 bg-emerald-50 text-emerald-800"
    : status === "PARTIAL"
      ? "border-amber-200 bg-amber-50 text-amber-900"
      : "border-slate-200 bg-slate-50 text-slate-600";
  return <span className={`rounded-full border px-3 py-1 text-xs ${style}`}>{status}</span>;
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl border border-slate-200 bg-white p-5"><p className="text-xs text-slate-500">{label}</p><p className="mt-2 text-lg font-semibold">{value}</p></div>;
}
