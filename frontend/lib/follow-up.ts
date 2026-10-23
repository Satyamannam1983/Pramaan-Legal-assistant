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
): FollowUpPrompt[] {
  const source = evidence.find((item) => item.document_id);
  if (!source) return [];

  const title = source.title?.trim() || source.document_id;
  return [
    {
      documentId: source.document_id,
      label: "Supporting evidence",
      query: `Which passage in ${title} most directly supports the answer?`,
    },
    {
      documentId: source.document_id,
      label: "Court's reasoning",
      query: `What legal reasoning does ${title} give on this issue?`,
    },
    {
      documentId: source.document_id,
      label: "Limits or exceptions",
      query: `What limitation or exception in ${title} could affect this answer?`,
    },
  ];
}
