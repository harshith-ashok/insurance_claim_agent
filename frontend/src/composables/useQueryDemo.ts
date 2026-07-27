import { computed, ref } from "vue";

export interface Chunk {
  src: string;
  page: number;
  section: string;
  score: number;
  verified: boolean;
  text: string;
}

export interface AnswerPart {
  text?: string;
  ref?: number;
}

export interface Scenario {
  query: string;
  chunks: Chunk[];
  answerParts: AnswerPart[];
  abstain?: boolean;
  trace: { embed: number; hybrid: number; rerank: number; gen: number };
}

export type Status = "idle" | "searching" | "generating" | "done";

// Talks to api/main.py (FastAPI, `uvicorn api.main:app`) — see CLAUDE.md.
const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

// Mirrors inference/guardrail.py's CITATION_RE and llm.py's ABSTENTION_MESSAGE —
// the frontend re-parses the raw answer text itself rather than have the
// backend ship pre-split spans, since both sides matching the same literal
// "(Page X, Section Y)" format is simpler than serializing offsets.
const CITATION_RE = /\(Page\s+(\d+),\s*Section\s+([^)]*)\)/gi;
const ABSTENTION_MESSAGE = "This information is not available in the provided policy documents.";

interface ApiChunkMeta {
  page: number;
  section: string;
  source: string;
  chunk_id: string;
}

interface ApiCandidate {
  text: string;
  rrf_score: number;
  cross_score: number | null;
  meta: ApiChunkMeta;
}

interface ApiCitationVerification {
  citation: { page: number; section: string };
  verified: boolean;
}

interface ApiTimings {
  embed_ms: number;
  hybrid_ms: number;
  rerank_ms: number;
  gen_ms: number;
}

interface ApiRetrievedPayload {
  candidates: ApiCandidate[];
  top_score: number | null;
  gated: boolean;
  timings: ApiTimings;
}

interface ApiDonePayload {
  answer: string | null;
  gated: boolean;
  abstained: boolean;
  timings: ApiTimings;
  context_candidates: ApiCandidate[];
  citation_verifications: ApiCitationVerification[];
}

function toChunks(candidates: ApiCandidate[], verifications: ApiCitationVerification[]): Chunk[] {
  return candidates.map((c) => {
    const forThisPage = verifications.filter((v) => v.citation.page === c.meta.page);
    const verified = forThisPage.length > 0 && forThisPage.every((v) => v.verified);
    return {
      src: c.meta.source,
      page: c.meta.page,
      section: c.meta.section,
      score: c.cross_score ?? 0,
      verified,
      text: c.text,
    };
  });
}

function toTrace(t: ApiTimings) {
  return { embed: t.embed_ms, hybrid: t.hybrid_ms, rerank: t.rerank_ms, gen: t.gen_ms };
}

// Splits the answer on (Page X, Section Y) tokens, resolving each to an
// index into `chunks` (by page) so AnswerPanel can render a clickable,
// scroll-to-source citation. A citation whose page isn't among the
// retrieved chunks (a real failure mode — see README's Known Limitations)
// is kept as plain, non-clickable text rather than dropped.
function toAnswerParts(answerText: string, chunks: Chunk[]): AnswerPart[] {
  const parts: AnswerPart[] = [];
  let lastIndex = 0;
  const re = new RegExp(CITATION_RE);
  let match: RegExpExecArray | null;
  while ((match = re.exec(answerText))) {
    if (match.index > lastIndex) {
      parts.push({ text: answerText.slice(lastIndex, match.index) });
    }
    const page = Number(match[1]);
    const chunkIdx = chunks.findIndex((c) => c.page === page);
    parts.push(chunkIdx !== -1 ? { ref: chunkIdx } : { text: match[0] });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < answerText.length || parts.length === 0) {
    parts.push({ text: answerText.slice(lastIndex) });
  }
  return parts;
}

const queryText = ref("");
const status = ref<Status>("idle");
const activeScenario = ref<Scenario | null>(null);
const activeChunk = ref<number | null>(null);
const answerMeta = ref("");
const streamingText = ref("");
const errorMessage = ref("");

const askDisabled = computed(() => status.value === "searching" || status.value === "generating");

const chunkMeta = computed(() => {
  if (status.value === "searching") return "searching…";
  if (status.value === "generating" || status.value === "done") {
    return activeScenario.value ? `top ${activeScenario.value.chunks.length} of 15` : "retrieving…";
  }
  return "";
});

let currentSource: EventSource | null = null;

function runQuery(text: string, filters?: { source?: string; section?: string }) {
  const trimmed = text.trim();
  if (!trimmed) return;

  currentSource?.close();

  queryText.value = trimmed;
  status.value = "searching";
  answerMeta.value = "";
  activeScenario.value = null;
  activeChunk.value = null;
  streamingText.value = "";
  errorMessage.value = "";

  const params = new URLSearchParams({ query: trimmed });
  if (filters?.source) params.set("target_source", filters.source);
  if (filters?.section) params.set("target_section", filters.section);

  const es = new EventSource(`${API_BASE}/query?${params.toString()}`);
  currentSource = es;

  es.addEventListener("retrieved", (e: MessageEvent) => {
    const data = JSON.parse(e.data) as ApiRetrievedPayload;
    activeScenario.value = {
      query: trimmed,
      chunks: toChunks(data.candidates, []),
      answerParts: [],
      trace: toTrace(data.timings),
    };
    status.value = "generating";
  });

  es.addEventListener("token", (e: MessageEvent) => {
    const data = JSON.parse(e.data) as { token: string };
    streamingText.value += data.token;
  });

  es.addEventListener("done", (e: MessageEvent) => {
    const data = JSON.parse(e.data) as ApiDonePayload;
    const answer = data.answer ?? "";
    const chunks = toChunks(data.context_candidates, data.citation_verifications);
    activeScenario.value = {
      query: trimmed,
      chunks,
      answerParts:
        data.abstained || answer === ABSTENTION_MESSAGE
          ? [{ text: answer }]
          : toAnswerParts(answer, chunks),
      abstain: data.abstained,
      trace: toTrace(data.timings),
    };
    answerMeta.value = data.abstained ? "abstained" : "gpt-oss:120b-cloud";
    status.value = "done";
    es.close();
    currentSource = null;
  });

  // Pipeline-level failure (Qdrant/LLM error) — named `query_error` rather
  // than the reserved `error` so it can't collide with EventSource's own
  // connection-failure event (handled below).
  es.addEventListener("query_error", (e: MessageEvent) => {
    const data = JSON.parse(e.data) as { error: string };
    errorMessage.value = data.error;
    status.value = "idle";
    es.close();
    currentSource = null;
  });

  es.addEventListener("error", () => {
    if (status.value === "done") return; // stream already finished cleanly
    errorMessage.value = `Couldn't reach the query API at ${API_BASE} — is \`uvicorn api.main:app\` running?`;
    status.value = "idle";
    es.close();
    currentSource = null;
  });
}

function focusChunk(i: number) {
  activeChunk.value = i;
}

export function useQueryDemo() {
  return {
    queryText,
    status,
    activeScenario,
    activeChunk,
    answerMeta,
    streamingText,
    errorMessage,
    askDisabled,
    chunkMeta,
    runQuery,
    focusChunk,
  };
}
