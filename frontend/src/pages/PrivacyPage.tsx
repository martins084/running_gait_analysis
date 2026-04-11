import { useI18n } from "../i18n";

export function PrivacyPage() {
  const { t } = useI18n();
  return (
    <div className="page">
      <section className="card">
        <h1 className="card__title">{t("privacy.title")}</h1>
        <p className="card__note">{t("privacy.summary")}</p>
        <ul>
          <li>{t("privacy.pointPurpose")}</li>
          <li>{t("privacy.pointRetention")}</li>
          <li>{t("privacy.pointRights")}</li>
          <li>{t("privacy.pointContact")}</li>
        </ul>
      </section>
    </div>
  );
}
