import { useCallback, useState } from "react";
import { useI18n } from "../i18n";

const ALLOWED = [".mp4", ".mov", ".avi", ".mkv"];
const MAX_MB = 200;

type Props = {
  disabled?: boolean;
  onFile: (file: File) => void;
};

export function UploadZone({ disabled, onFile }: Props) {
  const { t } = useI18n();
  const [drag, setDrag] = useState(false);

  const validateAndSend = useCallback(
    (file: File) => {
      const ext = file.name.toLowerCase().slice(file.name.lastIndexOf("."));
      if (!ALLOWED.includes(ext)) {
        alert(t("upload.badFormat", { list: ALLOWED.join(", ") }));
        return;
      }
      if (file.size > MAX_MB * 1024 * 1024) {
        alert(t("upload.tooLarge", { max: MAX_MB }));
        return;
      }
      onFile(file);
    },
    [onFile, t],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDrag(false);
      if (disabled) return;
      const f = e.dataTransfer.files[0];
      if (f) validateAndSend(f);
    },
    [disabled, validateAndSend],
  );

  return (
    <div
      className={`upload-zone ${drag ? "upload-zone--drag" : ""} ${disabled ? "upload-zone--disabled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
    >
      <p className="upload-zone__title">{t("upload.title")}</p>
      <p className="upload-zone__hint">
        {t("upload.hint", { formats: ALLOWED.join(", "), max: MAX_MB })}
      </p>
      <label className="upload-zone__btn">
        <input
          type="file"
          accept={ALLOWED.join(",")}
          disabled={disabled}
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) validateAndSend(f);
            e.target.value = "";
          }}
        />
        {t("upload.chooseFile")}
      </label>
    </div>
  );
}
