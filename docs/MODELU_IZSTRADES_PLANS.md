# Modeļu izstrādes detalizēts plāns

## 1) Mērķis un gala rezultāts

Šī dokumenta mērķis ir definēt pilnu, praktisku plānu, kā projektā izstrādāt un novērtēt divus ML virzienus:

1. **Gaitas fāžu noteikšanas modelis** (klasifikācija).
2. **Anomāliju noteikšanas modelis** (rekonstrukcijas kļūda / novelty detection).

Gala rezultātam jābūt:
- reproducējamai treniņa procedūrai,
- skaidram validācijas protokolam,
- salīdzinājumam ar atsauces mērījumiem,
- integrācijai API/produkcijas plūsmā.

---

## 2) Esošā projekta bāze (uz ko balstāmies)

Šobrīd repozitorijā jau ir:
- `models/gait_classifier.py` ar:
  - `GaitPhaseClassifier` (Conv1d + LSTM + FC),
  - `GaitAnomalyDetector` (LSTM encoder-decoder);
- `core/pose_detection.py` (MediaPipe pozu iegūšana),
- `core/feature_extractor.py` (pazīmju aprēķins no pozu secības),
- validācijas skripti:
  - `scripts/validate_features.py`,
  - `scripts/evaluate_ground_contact.py`.

Tas nozīmē, ka nav jāsāk “no nulles” — jāuzbūvē disciplinēta datu + treniņa + validācijas virskārta.

---

## 3) Datu avoti un to loma treniņā

## 3.1 Primārais avots fāžu modelim

**Biomehānikas datu kopa ar running saturu** (Nature/Scientific Data raksts un Figshare+ datu glabātuve) ir galvenais avots, jo:
- satur arī **skriešanas** sesijas,
- ir laboratorijas kvalitātes kinemātika,
- satur metadatus (ātrums, vecums, injury statuss utt.).

Lietojums:
- fāžu klasifikācijas modelim,
- benchmark/atsauces validācijai,
- kļūdu analīzei pa ātruma grupām.

## 3.2 Sekundārais avots robustumam

**Health&Gait**:
- izmantot reprezentāciju/pretraining robustumam,
- neizmantot kā vienīgo avotu “running fāžu” gala secinājumiem.

## 3.3 “In-the-wild” robustums

**Gait3D**:
- izmantot domēna noturības testiem / pretraining eksperimentiem,
- apzināties, ka primārais uzdevums tur ir gait recognition, nevis fāžu etiķetēšana.

---

## 4) Datu strukturēšana un split stratēģija

## 4.1 Obligātie principi

1. **Split pēc subjekta (`subject-level split`)**, nevis pēc video.
2. Nekādā gadījumā viens un tas pats subjekts nedrīkst būt train un val/test.
3. Saglabāt split failu versētu (`config/`), lai var reproducēt rezultātus.

## 4.2 Ieteiktais split

- Train: 70%
- Validation: 15%
- Test: 15%

Papildus:
- stratificēt pēc speed bucket (lēns/vidējs/ātrs),
- atsevišķa tabula injury vs non-injury sadalījumam.

## 4.3 Datu manifests

Izveidot vienotu CSV/JSON manifestu ar laukiem:
- `subject_id`
- `session_id`
- `video_path` / `sequence_path`
- `source_dataset` (Nature/Health&Gait/Gait3D)
- `mode` (walk/run)
- `speed_mps` (ja pieejams)
- `split` (train/val/test)
- `labels_available` (true/false)

---

## 5) Feature pipeline un input standarts

Modelim standarta ieeja:
- `(batch, seq_len, 66)` = 33 locītavas × 2 (x,y).

Soļi:
1. Pose extraction / import.
2. Joint mapping uz vienotu 33-locītavu shēmu.
3. Normalizācija (piem., pēc iegurņa/torso references).
4. Laika logi (`windowing`) ar overlap.
5. Kvalitātes filtri:
   - visibility slieksnis,
   - trūkstošo punktu apstrāde,
   - optional smoothing.

Mērķis: vienāds input formāts visiem datu avotiem.

---

## 6) Fāžu noteikšanas modeļa izstrāde

## 6.1 Arhitektūra (esošā bāze)

No `models/gait_classifier.py`:
- Conv1d(66→64),
- Conv1d(64→128),
- LSTM(128→hidden),
- FC + dropout + logits.

## 6.2 Treniņa konfigurācija (v1)

- Loss: CrossEntropyLoss.
- Optimizer: AdamW.
- Learning rate sākums: `1e-3` (scheduler ar samazināšanu pēc val-loss).
- Batch size: sākt ar 16 (pielāgot pēc VRAM/RAM).
- Epochs: 40–80 ar early stopping.
- Regularizācija: dropout + weight decay.

## 6.3 Metrikas

- Macro F1 (primārā),
- per-class F1,
- confusion matrix,
- balanced accuracy.

Papildus:
- metrika pa speed bucket,
- metrika pa skata tipu (ja pieejams),
- metrika pa datu avotu.

---

## 7) Anomāliju noteikšanas modeļa izstrāde

## 7.1 Arhitektūra (esošā bāze)

`GaitAnomalyDetector`:
- encoder LSTM,
- decoder LSTM,
- anomālijas score = rekonstrukcijas kļūda (MSE vai robust variants).

## 7.2 Treniņa režīms

- Treniņš uz “normālu” secību kopas.
- Validation laikā izvēlēties threshold:
  - ROC-based,
  - vai fixed false-positive target.

## 7.3 Metrikas

- AUROC,
- AUPRC,
- sensitivity/specificity pie izvēlētā sliekšņa.

---

## 8) Validācija pret atsauci (“gold/reference”)

Lai pierādītu metodikas vērtību:

1. Leņķu validācija:
   - `scripts/validate_features.py`,
   - MAE/RMSE/korelācija.
2. Ground-contact/fāžu notikumu validācija:
   - `scripts/evaluate_ground_contact.py`,
   - sensitivity/specificity/F1/ROC.
3. Kļūdas gradienta analīze pret ātrumu:
   - kļūda kā funkcija no speed,
   - regresija un grupu salīdzinājums.

Svarīgi:
- secinājumos teikt “atsauces salīdzinājums”, ja nav tieša MoCap-to-model frame-level pairing.

---

## 9) Eksperimentu matrica (ko obligāti izskriet)

## 9.1 Fāžu modelis

1. Baseline: tikai LSTM.
2. Kandidāts A: Conv1d + LSTM (esošais).
3. Kandidāts B: Conv1d + LSTM + cits window size.

Salīdzināt:
- macro F1,
- per-class F1,
- inference latency.

## 9.2 Domēna noturība

1. Train tikai primārajā running datu kopā.
2. Pretrain uz Health&Gait/Gait3D + fine-tune uz running.

Salīdzināt:
- test F1,
- robustumu pa speed bucket.

---

## 10) Integrācija produkta plūsmā

Pēc modeļa izvēles:
1. Saglabāt modeļa checkpoint + config hash.
2. Izveidot inferenci API ceļam:
   - optional flags, lai nepalēninātu pamatplūsmu.
3. Pievienot rezultātu struktūrai:
   - model_version,
   - confidence/per-frame classes.
4. Pievienot testus:
   - shape checks,
   - API response schema,
   - regressijas testi.

---

## 11) Veiktspējas un mērogojamības novērtējums

Jāmēra:
- apstrādes laiks uz 10s/30s/60s video,
- atkarība no FPS,
- CPU/RAM patēriņš (ar/bez ML inferenču).

Izeja:
- tabula ar throughput un latency,
- secinājumi par Docker/worker skaitu un praktisko deployment robežu.

---

## 12) Riski un mitigācija

1. **Domēna nobīde** (walking vs running):
   - risināt ar fine-tune uz running.
2. **Label quality**:
   - pārbaudīt inter-annotator vienošanos.
3. **Data leakage**:
   - strict subject-level split.
4. **Pose extractor atšķirības**:
   - standartizēt mapping + normalizāciju.

---

## 13) Darbu sadalījums: ko dari tu un ko izdarīšu es

## 13.1 Ko dari tu (nepieciešamais no tava puses)

1. Apstiprini datu avotus/licences un lejupielādes.
2. Nodrošini piekļuvi datu mapēm (lokālais ceļš).
3. Apstiprini mērķa metrikas (primārā = macro F1 vai cita).
4. Apstiprini, vai gala secinājumi fokusējas uz running-only.

## 13.2 Ko izdarīšu es (ieviešanas atbalsts)

1. Izveidošu datu manifesta ģeneratoru skriptu.
2. Izveidošu split generatoru ar subject-level stratifikāciju.
3. Sagatavošu treniņa skriptus:
   - fāžu modelim,
   - anomāliju modelim.
4. Pievienošu novērtēšanas skriptus un rezultātu tabulu eksportu.
5. Integrēšu modeļa inferenci API (ar konfigurējamu slodzi).
6. Uzrakstīšu reproducējamus “runbook” soļus (`docs/`).

---

## 14) Praktiska secība (nedēļu plāns)

## Nedēļa 1
- Datu manifests + split.
- Input pipeline validācija (shape, mapping).

## Nedēļa 2
- Fāžu modeļa baseline + kandidāts A.
- Pirmie metriķu grafiki.

## Nedēļa 3
- Anomāliju modelis + threshold tuning.
- Salīdzinājums pa speed bucket.

## Nedēļa 4
- Integrācija API + veiktspējas mērījumi.
- Gala tabulas, attēli, secinājumi darbam.

---

## 15) Definīcija “darbs izdarīts”

Darbs tiek uzskatīts par pabeigtu, ja ir:
- reproducējams treniņa ceļš (viens komandu komplekts no nulles),
- rezultātu tabula (val/test) ar primāro metriku,
- salīdzinājums ar atsauci un kļūdu analīze,
- API inferenču demonstrācija,
- dokumentēti ierobežojumi un nākamie uzlabojumi.

