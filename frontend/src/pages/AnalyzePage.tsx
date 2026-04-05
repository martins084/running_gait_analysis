import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { postAnalyzeVideo } from "../api/client";
import { UploadZone } from "../components/UploadZone";
import { useI18n } from "../i18n";

const LAST_ID_KEY = "gait-last-analysis-id";

export function AnalyzePage() {
  const { t } = useI18n();
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [lastId, setLastId] = useState<string | null>(() =>
    typeof sessionStorage !== "undefined" ? sessionStorage.getItem(LAST_ID_KEY) : null,
  );

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
    setError(null);
    setBusy(true);
    try {
      const res = await postAnalyzeVideo(file);
      sessionStorage.setItem(LAST_ID_KEY, res.id);
      setLastId(res.id);
      navigate(`/results/${res.id}`, { replace: false });
    } catch (e) {
      setError(e instanceof Error ? e.message : t("analyze.failedGeneric"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page page--analyze">
      {/* Vertikāli centrēts bloks — izmanto brīvo vietu zem augšējās joslas, nevis saspiest augšā */}
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
          </div>
          <div aria-busy={busy}>
            <UploadZone disabled={busy} onFile={handleFile} />
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
