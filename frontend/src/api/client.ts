import type { AnalyzeResponse, HealthResponse, PosesJson } from "./types";

/**
 * Base URL for the FastAPI server. Override with VITE_API_BASE_URL (no trailing slash).
 * Example: http://localhost:8000
 *
 * In **dev**, when `VITE_API_BASE_URL` is unset, returns `""` and `resolveApiUrl` uses the
 * Vite `/api` proxy (same origin) so `fetch()` does not depend on CORS — unlike `<video src>`,
 * which can load cross-origin without CORS and previously masked misconfigured origins.
 */
export function getApiBase(): string {
  const raw = import.meta.env.VITE_API_BASE_URL;
  if (raw && typeof raw === "string" && raw.trim()) {
    return raw.replace(/\/$/, "");
  }
  if (import.meta.env.DEV) {
    return "";
  }
  return "http://localhost:8000";
}

/** Resolve a path from the API (e.g. "/download/uuid") to a URL the browser can request. */
export function resolveApiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  const p = path.startsWith("/") ? path : `/${path}`;
  if (import.meta.env.DEV && !import.meta.env.VITE_API_BASE_URL?.trim()) {
    return `/api${p}`;
  }
  const base = getApiBase();
  return `${base}${p}`;
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const data = (await res.json()) as { detail?: unknown };
    const d = data.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return JSON.stringify(d);
    return res.statusText || `HTTP ${res.status}`;
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(resolveApiUrl("/health"));
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<HealthResponse>;
}

export async function postAnalyzeVideo(file: File): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("video", file, file.name);

  const res = await fetch(resolveApiUrl("/analyze"), {
    method: "POST",
    body: form,
  });

  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<AnalyzeResponse>;
}

export async function getResults(analysisId: string): Promise<AnalyzeResponse> {
  const res = await fetch(resolveApiUrl(`/results/${encodeURIComponent(analysisId)}`));
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<AnalyzeResponse>;
}

/** Full BlazePose JSON for an analysis (large). Used for client-side COM if needed. */
export async function getPosesJson(analysisId: string): Promise<PosesJson> {
  const res = await fetch(resolveApiUrl(`/poses/${encodeURIComponent(analysisId)}`));
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<PosesJson>;
}
