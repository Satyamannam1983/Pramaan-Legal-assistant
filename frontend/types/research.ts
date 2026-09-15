export type SourceType =
  | "act"
  | "central_law"
  | "state_law"
  | "rule"
  | "supreme_court_judgment"
  | "high_court_judgment"
  | "notification"
  | "government_document";

export interface SearchResult {
  id: string;
  title: string;
  citation?: string;
  snippet: string;
  source: string;
}
