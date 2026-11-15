import type {
  CitationRef,
  ComparisonCitation,
  SearchResult,
} from "./api";

export type WorkspaceMode = "general" | "document" | "compare";

export type ChatTurn = {
  id: string;
  question: string;
  answer: string;
  confidence?: string;
  generationMode?: string;
  citations: Array<CitationRef | ComparisonCitation>;
  sources: SearchResult[];
  abstained: boolean;
  limitations: string[];
  mode: WorkspaceMode;
};

export type AssistantSession = {
  mode: WorkspaceMode;
  query: string;
  followUp: boolean;
  followUpLabel: string | null;
  selectedDocument: string;
  comparisonDocuments: [string, string];
  conversationId: string | null;
  turns: ChatTurn[];
};

export const ASSISTANT_SESSION_KEY = "pramaan.assistant-session.v1";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isNullableNumber(value: unknown): value is number | null {
  return value === null || (typeof value === "number" && Number.isFinite(value));
}

function isCitation(value: unknown): value is CitationRef | ComparisonCitation {
  return isRecord(value)
    && typeof value.citation_id === "string"
    && typeof value.chunk_id === "string"
    && typeof value.document_id === "string"
    && isNullableString(value.section)
    && isNullableNumber(value.page_start)
    && isNullableNumber(value.page_end)
    && (value.title === undefined || isNullableString(value.title))
    && (value.source_url === undefined || isNullableString(value.source_url));
}

function isSearchResult(value: unknown): value is SearchResult {
  return isRecord(value)
    && typeof value.chunk_id === "string"
    && typeof value.content === "string"
    && typeof value.document_id === "string"
    && isNullableString(value.title)
    && isNullableString(value.section)
    && isNullableNumber(value.page_start)
    && isNullableNumber(value.page_end);
}

function isChatTurn(value: unknown): value is ChatTurn {
  return isRecord(value)
    && typeof value.id === "string"
    && typeof value.question === "string"
    && typeof value.answer === "string"
    && (value.confidence === undefined || typeof value.confidence === "string")
    && (value.generationMode === undefined || typeof value.generationMode === "string")
    && Array.isArray(value.citations)
    && value.citations.every(isCitation)
    && Array.isArray(value.sources)
    && value.sources.every(isSearchResult)
    && typeof value.abstained === "boolean"
    && Array.isArray(value.limitations)
    && value.limitations.every((item) => typeof item === "string")
    && (value.mode === "general" || value.mode === "document" || value.mode === "compare");
}

function isWorkspaceMode(value: unknown): value is WorkspaceMode {
  return value === "general" || value === "document" || value === "compare";
}

function isDocumentPair(value: unknown): value is [string, string] {
  return Array.isArray(value)
    && value.length === 2
    && value.every((item) => typeof item === "string");
}

export function readAssistantSession(storage: Pick<Storage, "getItem">): AssistantSession | null {
  const saved = storage.getItem(ASSISTANT_SESSION_KEY);
  if (!saved) return null;

  try {
    const value: unknown = JSON.parse(saved);
    if (
      !isRecord(value)
      || !isWorkspaceMode(value.mode)
      || typeof value.query !== "string"
      || (value.followUp !== undefined && typeof value.followUp !== "boolean")
      || (value.followUpLabel !== undefined && value.followUpLabel !== null && typeof value.followUpLabel !== "string")
      || typeof value.selectedDocument !== "string"
      || !isDocumentPair(value.comparisonDocuments)
      || !isNullableString(value.conversationId)
      || !Array.isArray(value.turns)
      || !value.turns.every(isChatTurn)
    ) {
      return null;
    }

    return {
      mode: value.mode,
      query: value.query,
      followUp: value.followUp === true,
      followUpLabel: typeof value.followUpLabel === "string" ? value.followUpLabel : null,
      selectedDocument: value.selectedDocument,
      comparisonDocuments: [value.comparisonDocuments[0], value.comparisonDocuments[1]],
      conversationId: value.conversationId,
      turns: value.turns,
    };
  } catch (error) {
    console.warn("Could not restore the saved assistant session.", error);
    return null;
  }
}

export function saveAssistantSession(
  storage: Pick<Storage, "setItem">,
  session: AssistantSession,
): void {
  try {
    storage.setItem(ASSISTANT_SESSION_KEY, JSON.stringify(session));
  } catch (error) {
    console.warn("Could not save the assistant session.", error);
  }
}
