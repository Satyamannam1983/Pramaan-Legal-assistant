"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { AppShell } from "../../components/AppShell";
import { PageHeader } from "../../components/PageHeader";
import {
  getErrorMessage,
  listDocuments,
  uploadDocument,
  type DocumentSummary,
  type UploadResponse,
} from "../../lib/api";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [loadingDocuments, setLoadingDocuments] = useState(true);
  const [documentsError, setDocumentsError] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const uploadController = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    listDocuments(controller.signal)
      .then(setDocuments)
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setDocumentsError(getErrorMessage(error));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingDocuments(false);
      });
    return () => {
      controller.abort();
      const activeUpload = uploadController.current;
      uploadController.current = null;
      activeUpload?.abort();
    };
  }, []);

  async function upload() {
    if (!selectedFile || uploading) return;
    setUploading(true);
    setUploadError("");
    setUploadResult(null);
    const controller = new AbortController();
    uploadController.current = controller;
    try {
      const result = await uploadDocument(selectedFile, controller.signal);
      if (uploadController.current !== controller) return;
      setUploadResult(result);
      setSelectedFile(null);
      const input = document.getElementById("document-upload") as HTMLInputElement | null;
      if (input) input.value = "";
      if (result.indexing_status === "indexed") {
        try {
          const refreshedDocuments = await listDocuments(controller.signal);
          if (uploadController.current === controller) setDocuments(refreshedDocuments);
          setDocumentsError("");
        } catch (error) {
          if (!controller.signal.aborted && uploadController.current === controller) {
            setDocumentsError(getErrorMessage(error));
          }
        }
      }
    } catch (error) {
      if (!controller.signal.aborted && uploadController.current === controller) {
        setUploadError(getErrorMessage(error));
      }
    } finally {
      if (uploadController.current === controller) {
        uploadController.current = null;
        setUploading(false);
      }
    }
  }

  return (
    <AppShell>
      <main className="page-container">
          <PageHeader
            eyebrow="Workspace / Documents"
            title="Documents"
            description="Upload, index, and review legal documents available to the research workspace."
            actions={<Link href="/" className="neutral-button">Open Legal Assistant</Link>}
          />

          <section className="mt-8 rounded-xl border border-slate-200 bg-white p-5">
            <h2 className="font-medium text-slate-900">Upload a document</h2>
            <p className="mt-1 text-sm text-slate-500">Supported formats: PDF, TXT, HTML, and JSON.</p>
            <div className="mt-4 flex flex-col gap-3 sm:flex-row">
              <input
                id="document-upload"
                type="file"
                accept=".pdf,.txt,.html,.htm,.json"
                disabled={uploading}
                onChange={(event) => {
                  setSelectedFile(event.target.files?.[0] ?? null);
                  setUploadResult(null);
                  setUploadError("");
                }}
                className="min-w-0 flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm"
              />
              <button
                type="button"
                onClick={upload}
                disabled={!selectedFile || uploading}
                className="rounded-lg bg-slate-900 px-5 py-2 text-sm font-medium text-white disabled:opacity-50"
              >
                {uploading ? "Uploading and indexing..." : "Upload"}
              </button>
            </div>
            {uploadError && <p role="alert" className="mt-3 rounded-lg bg-red-50 p-3 text-sm text-red-700">{uploadError}</p>}
            {uploadResult && (
              <div className={`mt-3 rounded-lg p-3 text-sm ${uploadResult.indexing_status === "indexed" ? "bg-emerald-50 text-emerald-800" : "bg-amber-50 text-amber-900"}`}>
                {uploadResult.indexing_status === "indexed" && (
                  <p>
                    <strong>{uploadResult.filename}</strong> was uploaded and indexed.
                    {uploadResult.document_id && (
                      <> <Link className="font-medium underline" href={`/documents/${encodeURIComponent(uploadResult.document_id)}`}>Ask about this document</Link>.</>
                    )}
                  </p>
                )}
                {uploadResult.indexing_status === "skipped" && (
                  <p><strong>{uploadResult.filename}</strong> was stored, but indexing was skipped. It will not be available for document search until indexed.</p>
                )}
                {uploadResult.indexing_status === "failed" && (
                  <p><strong>{uploadResult.filename}</strong> was stored, but indexing failed. It is not yet available for document search.</p>
                )}
                {uploadResult.detail && <p className="mt-1">{uploadResult.detail}</p>}
              </div>
            )}
          </section>

          {documentsError && <p role="alert" className="mt-6 rounded-lg bg-red-50 p-4 text-sm text-red-700">{documentsError}</p>}
          <div className="mt-8 overflow-hidden rounded-xl border border-slate-200 bg-white">
            {loadingDocuments ? (
              <p className="p-6 text-sm text-slate-500">Loading indexed documents...</p>
            ) : documents.map((doc) => (
              <Link key={doc.document_id} href={`/documents/${encodeURIComponent(doc.document_id)}`} className="block border-b border-slate-100 p-5 hover:bg-slate-50">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <h2 className="font-medium text-slate-900">{doc.title || doc.document_id}</h2>
                    <p className="mt-1 text-xs text-slate-500">{doc.document_type || "Unknown"} · {doc.court || "Court not specified"} · {doc.chunk_count} chunks</p>
                  </div>
                  <span className="text-xs text-slate-400">{doc.date || "Date unavailable"}</span>
                </div>
              </Link>
            ))}
            {!loadingDocuments && !documentsError && !documents.length && (
              <p className="p-6 text-sm text-slate-500">No indexed documents yet.</p>
            )}
          </div>
      </main>
    </AppShell>
  );
}
