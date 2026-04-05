import {
  createContext,
  useCallback,
  useContext,
  useLayoutEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { en } from "./locales/en";
import { lv } from "./locales/lv";

const LOCALE_KEY = "gait-ui-locale";

export type Locale = "lv" | "en";

const bundles = { en, lv } as const;

function readStoredLocale(): Locale {
  if (typeof window === "undefined") return "lv";
  const s = localStorage.getItem(LOCALE_KEY);
  if (s === "lv" || s === "en") return s;
  return "lv";
}

function getPath(obj: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((o, k) => {
    if (o !== null && typeof o === "object" && k in (o as object)) {
      return (o as Record<string, unknown>)[k];
    }
    return undefined;
  }, obj);
}

function interpolate(template: string, vars: Record<string, string | number>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_, key: string) =>
    key in vars ? String(vars[key as keyof typeof vars]) : "",
  );
}

type I18nContextValue = {
  locale: Locale;
  setLocale: (loc: Locale) => void;
  t: (path: string, vars?: Record<string, string | number>) => string;
};

const I18nContext = createContext<I18nContextValue | null>(null);

type Props = { children: ReactNode };

export function LocaleProvider({ children }: Props) {
  const [locale, setLocaleState] = useState<Locale>(() =>
    typeof window !== "undefined" ? readStoredLocale() : "lv",
  );

  const setLocale = useCallback((loc: Locale) => {
    setLocaleState(loc);
    localStorage.setItem(LOCALE_KEY, loc);
  }, []);

  useLayoutEffect(() => {
    document.documentElement.lang = locale === "lv" ? "lv" : "en";
    document.title = locale === "lv" ? "Skrējiena gaitas analīze" : "Running Gait Analysis";
  }, [locale]);

  const t = useCallback(
    (path: string, vars?: Record<string, string | number>) => {
      const raw = getPath(bundles[locale], path);
      if (typeof raw !== "string") return path;
      if (!vars) return raw;
      return interpolate(raw, vars);
    },
    [locale],
  );

  const value = useMemo(() => ({ locale, setLocale, t }), [locale, setLocale, t]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error("useI18n must be used within LocaleProvider");
  }
  return ctx;
}
