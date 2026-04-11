import { useCallback, useEffect, useLayoutEffect, useState, type ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import { getHealth } from "../api/client";
import { useI18n, type Locale } from "../i18n";

const THEME_KEY = "gait-ui-theme";

function readStoredTheme(): "light" | "dark" {
  const s = localStorage.getItem(THEME_KEY);
  if (s === "light" || s === "dark") return s;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

type Props = {
  children: ReactNode;
};

export function AppShell({ children }: Props) {
  const { pathname } = useLocation();
  const { t, locale, setLocale } = useI18n();
  /** On `/` the user is already starting an analysis — top “New analysis” would be a no-op duplicate of the brand. */
  const showTopNewAnalysis = pathname !== "/";
  const [theme, setTheme] = useState<"light" | "dark">(() =>
    typeof window !== "undefined" ? readStoredTheme() : "light",
  );
  const [health, setHealth] = useState<"ok" | "err" | "unk">("unk");

  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((th) => (th === "light" ? "dark" : "light"));
  }, []);

  const pickLocale = useCallback(
    (loc: Locale) => {
      setLocale(loc);
    },
    [setLocale],
  );

  useEffect(() => {
    let cancelled = false;
    const ping = () => {
      getHealth()
        .then(() => {
          if (!cancelled) setHealth("ok");
        })
        .catch(() => {
          if (!cancelled) setHealth("err");
        });
    };
    ping();
    const id = window.setInterval(ping, 45000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  return (
    <>
      <a href="#main-content" className="skip-link">
        {t("nav.skip")}
      </a>
      <header className="top-bar" role="banner">
        <div className="top-bar__inner">
          <Link to="/" className="top-bar__brand">
            {t("nav.brand")}
            <span className="top-bar__brand-sub">{t("nav.brandSub")}</span>
          </Link>
          {showTopNewAnalysis ? (
            <nav className="top-bar__nav" aria-label={t("nav.primaryAria")}>
              <Link className="top-bar__link" to="/">
                {t("nav.newAnalysis")}
              </Link>
              <Link className="top-bar__link" to="/privacy">
                {t("privacy.link")}
              </Link>
            </nav>
          ) : (
            <div className="top-bar__nav-spacer" aria-hidden="true" />
          )}
          <div className="top-bar__actions">
            <span
              className={["health-dot", health === "ok" ? "health-dot--ok" : "", health === "err" ? "health-dot--err" : ""]
                .filter(Boolean)
                .join(" ")}
              title={
                health === "ok" ? t("nav.apiOk") : health === "err" ? t("nav.apiErr") : t("nav.apiCheck")
              }
            >
              <span className="health-dot__mark" aria-hidden />
              {health === "ok" ? "API" : health === "err" ? "API" : "…"}
            </span>
            <div className="lang-toggle" role="group" aria-label={t("lang.aria")}>
              <button
                type="button"
                className={`lang-toggle__btn${locale === "lv" ? " lang-toggle__btn--active" : ""}`}
                onClick={() => pickLocale("lv")}
                aria-pressed={locale === "lv"}
              >
                {t("lang.lv")}
              </button>
              <button
                type="button"
                className={`lang-toggle__btn${locale === "en" ? " lang-toggle__btn--active" : ""}`}
                onClick={() => pickLocale("en")}
                aria-pressed={locale === "en"}
              >
                {t("lang.en")}
              </button>
            </div>
            <button
              type="button"
              className="theme-toggle"
              onClick={toggleTheme}
              aria-label={theme === "light" ? t("theme.ariaDark") : t("theme.ariaLight")}
            >
              {theme === "light" ? t("theme.dark") : t("theme.light")}
            </button>
          </div>
        </div>
      </header>
      <main id="main-content" className="app-main" role="main">
        {children}
      </main>
    </>
  );
}
