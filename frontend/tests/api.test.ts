import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import {
  ApiError,
  ApiTimeoutError,
  askDocument,
  askQuestion,
  compareDocuments,
  getEvaluationReport,
  requestJson,
  safeSourceUrl,
  uploadDocument,
} from "../lib/api";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

test("general question request uses the chat endpoint and query contract", async () => {
  let requestedUrl = "";
  let requestInit: RequestInit | undefined;
  globalThis.fetch = async (input, init) => {
    requestedUrl = String(input);
    requestInit = init;
    return Response.json({ answer: "ok" });
  };

  await askQuestion({ query: "  contract  " });

  assert.match(requestedUrl, /\/api\/chat$/);
  assert.equal(requestInit?.method, "POST");
  assert.deepEqual(JSON.parse(String(requestInit?.body)), { query: "  contract  " });
});

test("RAG requests allow time for an initial local embedding-model load", async () => {
  const originalSetTimeout = globalThis.setTimeout;
  let timeoutMs: number | undefined;
  globalThis.setTimeout = ((...args: Parameters<typeof setTimeout>) => {
    timeoutMs = args[1];
    return originalSetTimeout(...args);
  }) as typeof setTimeout;
  globalThis.fetch = async () => Response.json({ answer: "ok" });

  try {
    await askQuestion({ query: "question" });
    assert.equal(timeoutMs, 180_000);
  } finally {
    globalThis.setTimeout = originalSetTimeout;
  }
});

test("document question preserves an encoded document ID and query body", async () => {
  let requestedUrl = "";
  let requestBody: unknown;
  globalThis.fetch = async (input, init) => {
    requestedUrl = String(input);
    requestBody = JSON.parse(String(init?.body));
    return Response.json({ answer: "ok" });
  };

  await askDocument("doc/one", { query: "question" });

  assert.match(requestedUrl, /\/documents\/doc%2Fone\/ask$/);
  assert.deepEqual(requestBody, { query: "question" });
});

test("comparison sends exactly the selected identifiers in their selected order", async () => {
  let requestBody: unknown;
  globalThis.fetch = async (_input, init) => {
    requestBody = JSON.parse(String(init?.body));
    return Response.json({ document_ids: ["second", "first"] });
  };

  await compareDocuments({ document_ids: ["second", "first"] });

  assert.deepEqual(requestBody, { document_ids: ["second", "first"] });
});

test("upload sends the selected file as multipart form data", async () => {
  let requestedUrl = "";
  let requestInit: RequestInit | undefined;
  globalThis.fetch = async (input, init) => {
    requestedUrl = String(input);
    requestInit = init;
    return Response.json({
      document_id: "doc-1",
      filename: "notice.txt",
      content_type: "text/plain",
      size: 4,
      status: "accepted",
      indexing_status: "skipped",
      detail: null,
    });
  };

  await uploadDocument(new File(["text"], "notice.txt", { type: "text/plain" }));

  assert.match(requestedUrl, /\/documents\/upload$/);
  assert.equal(requestInit?.method, "POST");
  assert.ok(requestInit?.body instanceof FormData);
  const fileEntry = (requestInit?.body as FormData).get("file");
  assert.ok(fileEntry instanceof File);
  assert.equal(fileEntry.name, "notice.txt");
  assert.equal(new Headers(requestInit?.headers).has("Content-Type"), false);
});

test("server errors are converted to safe user messages", async () => {
  globalThis.fetch = async () =>
    Response.json({ detail: "secret database address" }, { status: 503 });

  await assert.rejects(
    requestJson("/chat"),
    (error: unknown) =>
      error instanceof ApiError &&
      error.status === 503 &&
      !error.message.includes("secret"),
  );
});

test("request timeout is reported distinctly", async () => {
  globalThis.fetch = async (_input, init) =>
    new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener(
        "abort",
        () => reject(new DOMException("Aborted", "AbortError")),
        { once: true },
      );
    });

  await assert.rejects(
    requestJson("/slow", {}, 1),
    (error: unknown) => error instanceof ApiTimeoutError,
  );
});

test("only HTTP(S) source links are rendered as links", () => {
  assert.equal(safeSourceUrl("https://example.test/source"), "https://example.test/source");
  assert.equal(safeSourceUrl("javascript:alert(1)"), null);
  assert.equal(safeSourceUrl("not a URL"), null);
});

test("evaluation API preserves stage statuses without inventing metrics", async () => {
  const report = {
    schema_version: 1,
    status: "NOT RUN",
    timestamp: "2026-01-01T00:00:00Z",
    dataset_size: 7,
    datasets: {},
    evaluations: {
      retrieval: { status: "NOT RUN", reason: "MongoDB is unavailable.", metrics: {} },
    },
    limitations: ["No retrieval corpus was available."],
  };
  globalThis.fetch = async () => Response.json(report);

  const response = await getEvaluationReport();

  assert.equal(response.dataset_size, 7);
  assert.equal(response.evaluations?.retrieval.status, "NOT RUN");
  assert.deepEqual(response.evaluations?.retrieval.metrics, {});
});
