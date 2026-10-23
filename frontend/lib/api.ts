export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";

const RAG_REQUEST_TIMEOUT_MS = 180_000;

export type CitationRef = {
  citation_id: string;
  chunk_id: string;
  document_id: string;
  title: string | null;
  section: string | null;
  subsection: string | null;
  page_start: number | null;
  page_end: number | null;
  document_type: string | null;
  court: string | null;
  date: string | null;
  source: string | null;
  source_url: string | null;
  relevant_quote: string | null;
};

export type ChatRequest = {
  query: string;
  conversation_id?: string | null;
};

export type ChatResponse = {
  answer: string;
  key_points: string[];
  citations: CitationRef[];
  limitations: string[];
  confidence: string;
  abstained: boolean;
  conversation_id: string;
  retrieval_attempts: number;
  strategies: string[];
  search_results: SearchResult[];
  generation_mode: string;
};

export type SearchRequest = {
  query: string;
  top_k?: number;
  document_id?: string | null;
  document_type?: string | null;
  court?: string | null;
  date_from?: string | null;
  date_to?: string | null;
};

export type SearchResult = {
  chunk_id: string;
  content: string;
  document_id: string;
  title: string | null;
  section: string | null;
  subsection: string | null;
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

export type SearchResponse = {
  query: string;
  results: SearchResult[];
};

export type DocumentQuestionRequest = {
  query: string;
};

export type DocumentSummary = {
  document_id: string;
  title: string | null;
  document_type: string | null;
  court: string | null;
  date: string | null;
  chunk_count: number;
  created_at: string | null;
};

export type DocumentComparisonRequest = {
  document_ids: [string, string];
};

export type ComparisonCitation = {
  citation_id: string;
  chunk_id: string;
  document_id: string;
  section: string | null;
  page_start: number | null;
  page_end: number | null;
  relevant_quote: string | null;
};

export type DocumentComparisonResponse = {
  document_ids: string[];
  overview: string;
  similarities: string[];
  differences: string[];
  legal_provisions: string[];
  reasoning_comparison: string[];
  outcome_comparison: string | null;
  citations: ComparisonCitation[];
  limitations: string[];
};

export type UploadResponse = {
  document_id: string | null;
  filename: string;
  content_type: string | null;
  size: number;
  status: string;
  indexing_status: "indexed" | "skipped" | "failed";
  detail: string | null;
};

export type DocumentDetail = {
  document_id: string;
  title: string | null;
  document_type: string | null;
  court: string | null;
  case_number: string | null;
  date: string | null;
  chunk_count: number;
};

export type DocumentAnalysisResponse = {
  document_id: string;
  summary: string;
  facts: string[];
  issues: string[];
  arguments: string[];
  reasoning: string[];
  decision: string | null;
  important_sections: string[];
  referenced_laws: string[];
  citations: ComparisonCitation[];
  limitations: string[];
};

export type EvaluationDatasetInfo = {
  name: string;
  version: string;
  provenance: string;
  example_count: number;
  sha256: string;
};

export type EvaluationStage = {
  status: string;
  reason?: string | null;
  metrics?: Record<string, unknown>;
  category_metrics?: Record<string, Record<string, unknown>>;
  operational_metrics?: Record<string, unknown>;
  diagnostics?: Record<string, unknown>;
  quality_evaluation?: {
    status: string;
    reason?: string | null;
    evaluator?: string | null;
    metrics?: Record<string, unknown>;
  };
  limitations?: string[];
};

export type EvaluationReport = {
  schema_version?: number;
  status: string;
  timestamp?: string;
  dataset_size: number;
  datasets?: Record<string, EvaluationDatasetInfo>;
  configuration?: Record<string, string | number | boolean | null>;
  evaluations?: Record<string, EvaluationStage>;
  limitations: string[];
};

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class ApiTimeoutError extends Error {
  constructor() {
    super("The request took too long. Please try again.");
    this.name = "ApiTimeoutError";
  }
}

function messageForStatus(status: number): string {
  if (status === 404) return "The requested document or resource was not found.";
  if (status === 413) return "The selected file exceeds the upload size limit.";
  if (status === 415) return "This file type is not supported.";
  if (status === 422) return "Please check the submitted information and try again.";
  if (status === 503) return "The service is temporarily unavailable. Please try again.";
  if (status >= 500) return "The server could not complete the request.";
  return "The request could not be completed.";
}

export async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = 30_000,
): Promise<T> {
  const controller = new AbortController();
  const callerSignal = init.signal;
  let timedOut = false;
  const abortFromCaller = () => controller.abort();
  if (callerSignal?.aborted) controller.abort();
  callerSignal?.addEventListener("abort", abortFromCaller, { once: true });
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    const response = await fetch(`${API_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });
    let data: unknown;
    try {
      data = await response.json();
    } catch (error) {
      if (timedOut || callerSignal?.aborted) throw error;
      data = null;
    }
    if (!response.ok) {
      throw new ApiError(messageForStatus(response.status), response.status);
    }
    if (data === null) {
      throw new ApiError("The server returned an unreadable response.", response.status);
    }
    return data as T;
  } catch (error) {
    if (timedOut) throw new ApiTimeoutError();
    if (callerSignal?.aborted) throw error;
    if (error instanceof ApiError || error instanceof ApiTimeoutError) throw error;
    throw new Error("Could not connect to the service. Check your connection and try again.");
  } finally {
    clearTimeout(timeout);
    callerSignal?.removeEventListener("abort", abortFromCaller);
  }
}

export function getErrorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "The request could not be completed. Please try again.";
}

function jsonRequest<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
  timeoutMs = 30_000,
) {
  return requestJson<T>(
    path,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    },
    timeoutMs,
  );
}

export function askQuestion(request: ChatRequest, signal?: AbortSignal) {
  return jsonRequest<ChatResponse>(
    "/chat",
    request,
    signal,
    RAG_REQUEST_TIMEOUT_MS,
  );
}

export function listDocuments(signal?: AbortSignal) {
  return requestJson<DocumentSummary[]>("/documents", { signal });
}

export function getDocument(documentId: string, signal?: AbortSignal) {
  return requestJson<DocumentDetail>(
    `/documents/${encodeURIComponent(documentId)}`,
    { signal },
  );
}

export function getDocumentAnalysis(documentId: string, signal?: AbortSignal) {
  return requestJson<DocumentAnalysisResponse>(
    `/documents/${encodeURIComponent(documentId)}/analysis`,
    { signal },
  );
}

export function askDocument(
  documentId: string,
  request: DocumentQuestionRequest,
  signal?: AbortSignal,
) {
  return jsonRequest<ChatResponse>(
    `/documents/${encodeURIComponent(documentId)}/ask`,
    request,
    signal,
    RAG_REQUEST_TIMEOUT_MS,
  );
}

export function compareDocuments(
  request: DocumentComparisonRequest,
  signal?: AbortSignal,
) {
  return jsonRequest<DocumentComparisonResponse>(
    "/documents/compare",
    request,
    signal,
    RAG_REQUEST_TIMEOUT_MS,
  );
}

export function uploadDocument(file: File, signal?: AbortSignal) {
  const body = new FormData();
  body.append("file", file);
  return requestJson<UploadResponse>(
    "/documents/upload",
    { method: "POST", body, signal },
    120_000,
  );
}

export function getEvaluationReport(signal?: AbortSignal) {
  return requestJson<EvaluationReport>("/evaluation", { signal });
}

export function safeSourceUrl(url: string | null): string | null {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:" ? parsed.href : null;
  } catch {
    return null;
  }
}
