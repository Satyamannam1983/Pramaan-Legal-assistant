import assert from "node:assert/strict";
import { test } from "node:test";
import {
  ASSISTANT_SESSION_KEY,
  readAssistantSession,
  saveAssistantSession,
  type AssistantSession,
} from "../lib/chat-session";

class MemoryStorage {
  private readonly values = new Map<string, string>();

  getItem(key: string) {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string) {
    this.values.set(key, value);
  }
}

const session: AssistantSession = {
  mode: "general",
  query: "A draft question",
  followUp: false,
  followUpLabel: null,
  selectedDocument: "doc-1",
  comparisonDocuments: ["doc-1", "doc-2"],
  conversationId: "conversation-1",
  turns: [
    {
      id: "turn-1",
      question: "What does this source say?",
      answer: "A concise answer. [SOURCE 1]",
      citations: [
        {
          citation_id: "SOURCE 1",
          chunk_id: "chunk-1",
          document_id: "doc-1",
          title: "Source",
          section: null,
          subsection: null,
          page_start: 2,
          page_end: 2,
          document_type: null,
          court: null,
          date: null,
          source: null,
          source_url: null,
          relevant_quote: null,
        },
      ],
      sources: [],
      abstained: false,
      limitations: [],
      mode: "general",
    },
  ],
};

test("assistant session survives page navigation and reloads", () => {
  const storage = new MemoryStorage();

  saveAssistantSession(storage, session);

  assert.deepEqual(readAssistantSession(storage), session);
});

test("invalid saved assistant sessions are ignored", () => {
  const storage = new MemoryStorage();
  storage.setItem(ASSISTANT_SESSION_KEY, JSON.stringify({ turns: ["invalid"] }));

  assert.equal(readAssistantSession(storage), null);
});

test("malformed saved assistant sessions are ignored", () => {
  const storage = new MemoryStorage();
  storage.setItem(ASSISTANT_SESSION_KEY, "{");

  assert.equal(readAssistantSession(storage), null);
});
