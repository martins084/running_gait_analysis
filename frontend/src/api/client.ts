import type { AnalyzeResponse, AuthResponse, DsarResponse, HealthResponse, PosesJson } from "./types";

const TOKEN_KEY = "gait-auth-token";

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

export function setAuthToken(token: string | null): void {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
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

async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers ?? {});
  const token = getAuthToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return fetch(resolveApiUrl(path), { ...init, headers });
}

/** Binary GET with auth — for `<video>` / downloads when Bearer cannot be sent via plain URL. */
export async function fetchAuthenticatedBlob(path: string): Promise<Blob> {
  const res = await apiFetch(path);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.blob();
}

export async function register(email: string, password: string): Promise<AuthResponse> {
  const res = await apiFetch("/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<AuthResponse>;
}

export async function login(email: string, password: string): Promise<AuthResponse> {
  const res = await apiFetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<AuthResponse>;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(resolveApiUrl("/health"));
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<HealthResponse>;
}

export async function postAnalyzeVideo(
  file: File,
  consentLocale: string,
  storageProfile: "minimal" | "standard" | "full" = "minimal",
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("video", file, file.name);
  form.append("consent_accepted", "true");
  form.append("consent_version", "v1");
  form.append("consent_timestamp", new Date().toISOString());
  form.append("consent_locale", consentLocale);
  form.append("storage_profile", storageProfile);

  const res = await apiFetch("/analyze", { method: "POST", body: form });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<AnalyzeResponse>;
}

export async function getResults(analysisId: string): Promise<AnalyzeResponse> {
  const res = await apiFetch(`/results/${encodeURIComponent(analysisId)}`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<AnalyzeResponse>;
}

export async function getPosesJson(analysisId: string): Promise<PosesJson> {
  const res = await apiFetch(`/poses/${encodeURIComponent(analysisId)}`);
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<PosesJson>;
}

export async function deleteAnalysis(analysisId: string): Promise<void> {
  const res = await apiFetch(`/analyses/${encodeURIComponent(analysisId)}`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reason: "user_requested" }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
}

export async function requestDsarExport(analysisId?: string): Promise<DsarResponse> {
  const res = await apiFetch("/dsar/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis_id: analysisId ?? null, reason: "data_subject_request" }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<DsarResponse>;
}

export async function requestDsarDelete(analysisId?: string): Promise<DsarResponse> {
  const res = await apiFetch("/dsar/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis_id: analysisId ?? null, reason: "data_subject_request" }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json() as Promise<DsarResponse>;
}
