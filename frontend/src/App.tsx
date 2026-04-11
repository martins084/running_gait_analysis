import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { AnalyzePage } from "./pages/AnalyzePage";
import { PrivacyPage } from "./pages/PrivacyPage";
import { ResultsPage } from "./pages/ResultsPage";

export default function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<AnalyzePage />} />
          <Route path="/results/:id" element={<ResultsPage />} />
          <Route path="/privacy" element={<PrivacyPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}
