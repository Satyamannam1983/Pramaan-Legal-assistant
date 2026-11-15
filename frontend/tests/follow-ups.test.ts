import assert from "node:assert/strict";
import { test } from "node:test";
import { buildFollowUpPrompts } from "../lib/follow-up";

test("follow-up prompts are absent without a traceable evidence document", () => {
  assert.deepEqual(buildFollowUpPrompts([], "An answer."), []);
});

test("three distinct follow-up prompts are scoped to the cited document", () => {
  const answer = "The court held that the agreement was enforceable.";
  const prompts = buildFollowUpPrompts([
    { document_id: "case-1", title: "A v. B" },
  ], answer);

  assert.equal(prompts.length, 3);
  assert.equal(new Set(prompts.map((prompt) => prompt.query)).size, 3);
  assert.ok(prompts.every((prompt) => prompt.documentId === "case-1"));
  assert.ok(prompts.every((prompt) => prompt.query.includes(answer)));
  assert.ok(prompts.every((prompt) => !prompt.query.includes("A v. B")));
  assert.ok(prompts.every((prompt) => !prompt.query.includes("case-1")));
  assert.deepEqual(prompts.map((prompt) => prompt.label), [
    "Show evidence",
    "Explain reasoning",
    "Find exceptions",
  ]);
  assert.ok(prompts.every((prompt) => !prompt.label.includes(answer)));
  assert.ok(prompts.every((prompt) => !prompt.label.includes("?")));
});

test("follow-up prompts do not expose a document identifier when no title is available", () => {
  const prompts = buildFollowUpPrompts(
    [{ document_id: "case-2", title: null }],
    "The court ordered a refund.",
  );

  assert.equal(prompts.length, 3);
  assert.ok(prompts.every((prompt) => !prompt.query.includes("case-2")));
});
