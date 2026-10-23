import assert from "node:assert/strict";
import { test } from "node:test";
import { buildFollowUpPrompts } from "../lib/follow-up";

test("follow-up prompts are absent without a traceable evidence document", () => {
  assert.deepEqual(buildFollowUpPrompts([]), []);
});

test("three distinct follow-up prompts are scoped to the cited document", () => {
  const prompts = buildFollowUpPrompts([
    { document_id: "case-1", title: "A v. B" },
  ]);

  assert.equal(prompts.length, 3);
  assert.equal(new Set(prompts.map((prompt) => prompt.query)).size, 3);
  assert.ok(prompts.every((prompt) => prompt.documentId === "case-1"));
  assert.ok(prompts.every((prompt) => prompt.query.includes("A v. B")));
});

test("follow-up prompts use the document ID when no title is available", () => {
  const prompts = buildFollowUpPrompts([{ document_id: "case-2", title: null }]);

  assert.equal(prompts.length, 3);
  assert.ok(prompts.every((prompt) => prompt.query.includes("case-2")));
});
