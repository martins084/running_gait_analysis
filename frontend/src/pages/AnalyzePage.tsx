import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { getAuthToken, login, postAnalyzeVideo, register, setAuthToken } from "../api/client";
import { UploadZone } from "../components/UploadZone";
import { useI18n } from "../i18n";

const LAST_ID_KEY = "gait-last-analysis-id";

export function AnalyzePage() {
  const { t, locale } = useI18n();
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [consent, setConsent] = useState(false);
  const [storageProfile, setStorageProfile] = useState<"minimal" | "standard" | "full">("minimal");
  const [lastId, setLastId] = useState<string | null>(() =>
    typeof sessionStorage !== "undefined" ? sessionStorage.getItem(LAST_ID_KEY) : null,
  );
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authed, setAuthed] = useState<boolean>(() => Boolean(getAuthToken()));

  useEffect(() => {
    if (!busy) return;
    setElapsed(0);
    const t0 = Date.now();
    const id = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - t0) / 1000));
    }, 1000);
    return () => window.clearInterval(id);
  }, [busy]);

  function formatElapsed(totalSec: number): string {
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return m > 0 ? t("analyze.timeMinSec", { m, s }) : t("analyze.timeSec", { s });
  }

  async function handleFile(file: File) {
    if (!authed) {
      setError(t("auth.needLogin"));
      return;
    }
    if (!consent) {
      setError(t("upload.consentRequired"));
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const res = await postAnalyzeVideo(file, locale, storageProfile);
      sessionStorage.setItem(LAST_ID_KEY, res.id);
      setLastId(res.id);
      navigate(`/results/${res.id}`, { replace: false });
    } catch (e) {
      setError(e instanceof Error ? e.message : t("analyze.failedGeneric"));
    } finally {
      setBusy(false);
    }
  }

  async function onLogin(registerMode: boolean) {
    setError(null);
    try {
      const response = registerMode ? await register(email, password) : await login(email, password);
      setAuthToken(response.token);
      setAuthed(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : t("auth.failed"));
    }
  }

  return (
    <div className="page page--analyze">
      <div className="analyze-page__stage">
        <div className="analyze-layout">
          <div className="analyze-layout__intro">
            <header className="hero">
              <h1 className="hero__title">{t("analyze.title")}</h1>
              <p className="hero__subtitle">{t("analyze.subtitle")}</p>
              <p className="hero__meta">{t("analyze.meta")}</p>
            </header>
            {lastId && (
              <Link className="resume-link link-back" to={`/results/${lastId}`}>
                {t("analyze.resumeLast")}
              </Link>
            )}
            <Link className="resume-link link-back" to="/privacy">
              {t("privacy.link")}
            </Link>
          </div>
          <div aria-busy={busy} className="analyze-layout__controls">
            {!authed && (
              <section className="card analyze-auth-card">
                <h2 className="card__title">{t("auth.title")}</h2>
                <p className="card__note">{t("auth.note")}</p>
                <div className="analyze-field-stack">
                  <input
                    className="input-control"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder={t("auth.email")}
                    autoComplete="email"
                  />
                  <input
                    className="input-control"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder={t("auth.password")}
                    type="password"
                    autoComplete="current-password"
                  />
                </div>
                <div className="analyze-auth-actions">
                  <button type="button" className="upload-zone__btn" onClick={() => onLogin(false)}>
                    {t("auth.login")}
                  </button>
                  <button type="button" className="button-secondary" onClick={() => onLogin(true)}>
                    {t("auth.register")}
                  </button>
                </div>
              </section>
            )}
            <section className="card analyze-settings-card">
              <label className="analyze-label">
                <span className="analyze-label__text">{t("upload.storageProfile")}</span>
                <select
                  className="select-control"
                  value={storageProfile}
                  onChange={(e) => setStorageProfile(e.target.value as "minimal" | "standard" | "full")}
                >
                  <option value="minimal">{t("upload.profileMinimal")}</option>
                  <option value="standard">{t("upload.profileStandard")}</option>
                  <option value="full">{t("upload.profileFull")}</option>
                </select>
              </label>
              <label className="analyze-consent">
                <input
                  className="checkbox-control"
                  type="checkbox"
                  checked={consent}
                  onChange={(e) => setConsent(e.target.checked)}
                />
                <span>{t("upload.consentLabel")}</span>
              </label>
            </section>
            <UploadZone disabled={busy || !consent || !authed} onFile={handleFile} />
          </div>
        </div>
      </div>

      {busy && (
        <div className="panel panel--progress" role="status" aria-live="polite">
          <div className="progress-indeterminate" aria-hidden />
          <div>
            <p className="panel__text">{t("analyze.processing")}</p>
            <p className="panel__timer">
              {t("analyze.elapsed")} {formatElapsed(elapsed)}
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="panel panel--error" role="alert">
          <strong>{t("analyze.errorPrefix")}</strong> {error}
        </div>
      )}
    </div>
  );
}
