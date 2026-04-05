/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  /** When "true", show mock ML gait phase timeline in development only. */
  readonly VITE_DEV_ML_MOCK?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
