import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
/* Latin Extended coverage for LV/EN with requested industrial type stack */
import "@fontsource/dm-mono/latin-ext-400.css";
import "@fontsource/dm-mono/latin-ext-500.css";
import "@fontsource/dm-sans/latin-ext-300.css";
import "@fontsource/dm-sans/latin-ext-400.css";
import "@fontsource/dm-sans/latin-ext-700.css";
import App from "./App";
import { LocaleProvider } from "./i18n";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/components.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <LocaleProvider>
      <App />
    </LocaleProvider>
  </StrictMode>,
);
