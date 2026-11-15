export type EvidenceReference = {
  document_id: string;
  title?: string | null;
};

export type FollowUpPrompt = {
  documentId: string;
  label: string;
  query: string;
};

export function buildFollowUpPrompts(
  evidence: readonly EvidenceReference[],
  answer: string,
): FollowUpPrompt[] {
  const source = evidence.find((item) => item.document_id);
  if (!source) return [];

  const answerContext = answer.trim().slice(0, 500);
  if (!answerContext) return [];
  return [
    {
      documentId: source.document_id,
      label: "Show evidence",
      query: `What evidence in the cited source supports this answer: "${answerContext}"?`,
    },
    {
      documentId: source.document_id,
      label: "Explain reasoning",
      query: `What reasoning in the cited source supports this answer: "${answerContext}"?`,
    },
    {
      documentId: source.document_id,
      label: "Find exceptions",
      query: `Does the cited source describe any exception to this answer: "${answerContext}"?`,
    },
  ];
}
