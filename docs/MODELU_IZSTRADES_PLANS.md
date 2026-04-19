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



## 16) Pašreizējais infrastruktūras statuss (DigitalOcean)

Šī sadaļa fiksē **reāli izpildīto** setup secību, lai vēlāk nav jāpārkonstruē, kas tieši tika darīts.

### 16.1 Datu avots (primārais)

- Figshare+ datu kopa: `https://doi.org/10.25452/figshare.plus.24255795`
- Galvenais arhīvs: `ric_data.zip` (aptuveni 22.75 GB)
- Pēc ekstrakcijas iegūta mape: `reformat_data/`
- Ekstrakcijas rezultātā verificēti `1798` subjekti (folder count)

### 16.2 Izmantotā compute/storage konfigurācija

- CPU droplet (ingest/preprocess): `ubuntu-s-1vcpu-2gb-fra1-01`
- Reģions: `FRA1`
- Bāzes disks: `50 GB` (nepietiek pilnai ekstrakcijai)
- Papildus pievienots Volumes Block Storage: `100 GB`
- Volume mountpoint: `/mnt/volume_fra1_01`

Praktisks secinājums:
- ekstrakcija uz root (`/`) nav droša ar 50 GB disku;
- ekstrakcija jāveic uz pievienotā volume mountpoint.

### 16.3 Izpildītā datu ingest secība

1. Lejupielādēts arhīvs no tiešā Figshare download endpoint:
   - `https://ndownloader.figshare.com/files/42651352`
2. Arhīvs pārvietots uz volume:
   - `/mnt/volume_fra1_01/ric_data.zip`
3. Ekstrakcija veikta uz volume:
   - mērķa mape: `/mnt/volume_fra1_01/reformat_data`
4. Verifikācija:
   - nav aktīvu `7z`/`unzip` procesu;
   - `reformat_data` satur `1798` ierakstus/subjektus.

### 16.4 Spaces konfigurācija (persistents datu slānis)

- Spaces bucket: `bakalaurs`
- Reģions/endpoint: `fra1`
- Origin endpoint: `https://bakalaurs.fra1.digitaloceanspaces.com`
- CLI endpoint (`--endpoint-url`): `https://fra1.digitaloceanspaces.com`

Svarīgi:
- `aws s3` komandās bucket tiek dots kā `s3://bakalaurs/...`;
- endpoint tiek dots atsevišķi kā reģiona endpoint, nevis obligāti bucket host.

### 16.5 Upload mērķa struktūra Spaces pusē

- `s3://bakalaurs/processed/reformat_data/`
- nākamajiem eksperimentiem rezervēt:
  - `s3://bakalaurs/raw/`
  - `s3://bakalaurs/checkpoints/`
  - `s3://bakalaurs/logs/`

### 16.6 Operacionālā stratēģija izmaksu kontrolei

1. CPU droplet tiek lietots tikai:
   - download,
   - extract,
   - sākotnējā datu sakārtošana,
   - augšupielāde uz Spaces.
2. GPU droplet tiek iedarbināts tikai tad, kad:
   - dati jau ir Spaces,
   - train skripti/config ir gatavi,
   - var sākt treniņu bez papildu setup kavēšanās.
3. Pēc pabeigta etapa:
   - izdzēst pagaidu lokālos failus,
   - apturēt/iznīcināt nevajadzīgos dropletus.

### 16.7 Nākamie tiešie soļi pēc upload pabeigšanas

1. Verificēt Spaces saturu (failu skaits + kopējais izmērs).
2. Izveidot `manifest` failu no `reformat_data` struktūras.
3. Definēt `subject-level split` (train/val/test).
4. Sagatavot pirmo baseline treniņa skriptu (CPU smoke test + GPU full run).

**Statusa atjauninājums:** punkti 2–4 ir **īstenoti kodā** (manifesta un splitu ģeneratori, anomāliju treniņa/novērtēšanas skripti, konfigurācija). Detalizēts apraksts: §17.5. Pilnam GPU treniņam joprojām jāseko §18.

## 17) Progress žurnāls (izdarīts līdz šim)

Šis ir īss “fakts -> rezultāts” žurnāls par jau paveikto, lai var izmantot bakalaura darba metodoloģijas aprakstā.

### 17.1 Infrastruktūra un glabāšana

- Izveidots CPU droplet (`ubuntu-s-1vcpu-2gb-fra1-01`) ingest etapam.
- Konstatēts, ka 50 GB root disks nav pietiekams pilnai ekstrakcijai.
- Pievienots `100 GB` Block Storage volume (`/mnt/volume_fra1_01`).
- Izveidots Spaces bucket `bakalaurs` reģionā `fra1`.

### 17.2 Datu ingest un ekstrakcija

- Lejupielādēts `ric_data.zip` no Figshare tiešā endpoint (`files/42651352`).
- Arhīvs pārvietots uz volume un veikta ekstrakcija `reformat_data` mapē.
- Verifikācija pēc ekstrakcijas:
  - nav aktīvu ekstrakcijas procesu;
  - mapē iegūti `1798` subjekti.

### 17.3 Upload uz Spaces

- Konfigurētas Spaces access atslēgas (`aws configure`, reģions `fra1`).
- Izpildīts rekursīvs upload uz:
  - `s3://bakalaurs/processed/reformat_data/`
- Spaces pusē verificēts saturs:
  - `2506` objekti;
  - kopējais izmērs aptuveni `44.9 GiB`.

### 17.4 Satura validācija no Spaces

- Veikts tiešs `aws s3 cp ... -` tests no viena JSON objekta:
  - `processed/reformat_data/100001/20110531T161051.json`
- Apstiprināts, ka JSON ir pieejams un strukturēts korekti.
- Konkrētajā paraugā:
  - `hz_r = 200`,
  - `running` sadaļa aizpildīta,
  - `walking` tukša (run-only sesija, kas ir sagaidāms šai datu kopai).

### 17.5 Repozitorijā ieviestā anomāliju (RIC) priekš-GPU plūsma

Lai bakalaura darbā var reproducējami aprakstīt ceļu no “raw JSON” līdz pirmajam treniņam, šeit fiksēts **faktiski implementētais** kods un faili (ne tikai plāns).

**Manifests un spliti**

- `scripts/build_ric_manifest.py` — skenē `reformat_data` struktūru (vai S3 prefiksu) un veido `data/processed/ric_manifest.csv` (+ summary JSON). Sesiju kopskaits praksē sakrīt ar publicēto kopkopu (~2506 sesijas pret ~1798 subjektiem).
- `scripts/build_subject_splits.py` — **subjektu līmeņa** sadalījums train/val/test (noklusējums 70/15/15, determinēts ar `seed`), validācija pret **datu noplūdi** starp splitiem; izvades faili:
  - `data/processed/splits/subject_split_v1.csv`
  - `data/processed/splits/session_split_v1.csv` (katrai sesijai mantota subjekta split vērtība).

**Datu ielāde un priekšapstrāde (metodoloģiski svarīgi)**

- `core/ric_dataset.py` — `RICAnomalyDataset` + `SequenceSpec`:
  - ieeja: lokāli JSON faili zem `json_root` (parasti `data/ric/reformat_data/...`), ceļš no manifesta `source_ref`;
  - izvēle starp `running` / `walking` blokiem pēc `prefer_mode` (`run` | `walk` | `auto`);
  - markeru vārdnīca → **deterministiska secība** (`sorted(marker_names)`), tad concatenate uz mātricu formā `[T, M·3]`;
  - **logu garums** `seq_len` (konfigā, piem. `180`): treniņā nejauša loga pozīcija, validācijā/testā centrēts logs;
  - **normalizācija**: loga ietvaros per-feature z-score (`normalize: zscore`).
- **Heterogēns markeru skaits starp sesijām** (darbā jāpiemin kā metodoloģisks ierobežojums / risinājums):
  - dažādās sesijās atšķiras `M`, tātad pēdējās dimensijas platums atšķiras;
  - `compute_max_feature_dim(...)` nosaka globālo maksimumu `D_max = max_s(M_s·3)` pēc split CSV un JSON;
  - katrā parauga pazīmes **tiek papildinātas ar nullēm** līdz `D_max`, garākas — **apgrieztas** līdz `D_max`, lai `DataLoader` varētu `stack` batch dimension;
  - tas dod fiksētu `input_size` autoenkoderim visā eksperimentā.

**Treniņš un novērtēšana**

- `config/anomaly_train_v1.yaml` — datu ceļi (`session_split_csv`, `json_root`), `seq_len`, režīms, vai treniņā iekļaut `is_injured` (noklusējumā treniņš tikai uz “veseliem” — `train_include_injured: false`), validācijā iekļaut arī injured (`val_include_injured: true`), hipotēze: novelty uz “normālu” gaitu.
- `scripts/train_anomaly.py` — `GaitAnomalyDetector` (LSTM encoder–decoder no `models/gait_classifier.py`), zaudējums = rekonstrukcijas MSE, AdamW, gradient clipping; saglabā `last.pt` / `best.pt`, metadatus (`run_metadata.json`, `epoch_metrics.csv`, `train_summary.json`) ar config hash un git commit, ja pieejams.
- `scripts/evaluate_anomaly.py` — rekonstrukcijas kļūda kā skaits, kvantīļu slieksnis, AUROC/AUPRC (ja abas klases klāt); izvade: `metrics.json`, `thresholds.json`, `per_sample_scores.csv`; **novērtēšanā** `input_size` ielasīts no checkpoint, lai sakristu ar treniņu.

**Palīgrīki un dokumentācija**

- `scripts/pre_gpu_check.py` — “viss uz ceļa” pārbaudes skripts pirms dārgā GPU laika.
- `scripts/sync_run_artifacts.py` — artefaktu sinhronizācija uz Spaces (dry-run režīms pārbaudāms pirms upload).
- `docs/AMS3_PRE_GPU_RUNBOOK.md` — angliski, soli pa solim: manifests → spliti → CPU smoke train → eval → go/no-go.

**Importu arhitektūra (lai CPU treniņš neprasītu OpenCV/MediaPipe)**

- `core/__init__.py` — samazināts “eager” imports: `from core import ric_dataset` vairs nedzen iekšā `cv2` / `mediapipe`, kas citādi lauza vidi, kurā ir tikai PyTorch + datu bibliotēkas. Bakalaura darba tekstā var piebilst, ka pose pipeline un RIC kinemātikas pipeline ir **atdalīti moduļi**.

### 17.6 DigitalOcean FRA CPU droplets: reālais izpildes konteksts (smoke treniņš)

- Repozitorijs uz volume: `/mnt/volume_fra1_01/work/running_gait_analysis` (vai līdzīgs ceļš).
- Vides Python: `python3` + projekta `.venv`; komandā izmantot `source .venv/bin/activate` vai tiešu `.venv/bin/python`.
- Lielie JSON paliek uz volume: bieži izmanto **simbolisko saiti** `data/ric/reformat_data` → `/mnt/volume_fra1_01/reformat_data`, lai YAML ceļš paliktu relatīvs pret repozitoriju.
- Metadatu CSV (Figshare / AMS `meta/`) var turēt lokāli zem `data/figshare/` vai lejupielādēt no `bakalaurs-ams`, lai būtu pieejami injury un citi lauki manifesta un splitu būvēšanā — atkarībā no tā, kā manifestu būvē.
- **Ilgstoši procesi**: SSH sesijas var pārtraukt; treniņš jāpalaiž **`tmux` vai `screen`** sesijā, atslēgties ar `Ctrl+B`, tad `D`; pēc tam `tmux attach -t <vārds>`. Ieteicams `python -u` nebuferizētai izdrukai.
- **Resursi**: 1 vCPU / 2 GB RAM — pilnam daudzu epoku treniņam tas ir lēns un OOM risks; smoke testam (`--epochs 1`) un metodikas validācijai tas ir pieņemami; pilnam eksperimentam paredzēts GPU (sk. §18).

### 17.7 Versiju kontrole (nozīmīgi commit darba reproducējamībai)

- `core/__init__.py` — samazināts smags imports, lai RIC pipeline darbotos “slim” vidē (piem., commit `34f2ada` uz `master`).
- `core/ric_dataset.py` + `scripts/train_anomaly.py` + `scripts/evaluate_anomaly.py` — vienots `feature_dim` / `input_size` risinājums heterogēnam markeru skaitam (piem., commit `a4477a9` uz `master`).

Precīzus hash ieteicams vienreiz pārbaudīt ar `git log -1` brīdī, kad raksti metodoloģijas nodaļu.

### 17.8 Ko šo bloku tieši izmantot bakalaura darba tekstā

- **Datu sadalījums**: subjektu līmenī, sesijas mantojumā — novērš identitātes noplūdi.
- **Uzdevuma formulējums**: anomālija kā **augstas rekonstrukcijas kļūdas** novirze no autoenkodera, apmācīta uz izvēlētu “references” kopu (šeit: opcija bez injured treniņā).
- **Tehniskā īpatnība datu kopai**: atšķirīgs optisko markeru skaits starp sesijām → fiksēta ieejas dimensija ar padding/truncate un globālo `D_max` — jāapraksta kā **pielāgošanās reālai MoCap heterogenitātei**, ne kā kļūda datu ielasē.
- **Reproducējamība**: YAML + split CSV versija + `seed` + checkpoint ar `input_size` / `hidden_size`.

## 18) Latest status snapshot un nākamā izpildes fāze

### 18.1 Latest status snapshot (apstiprināts)

- FRA Spaces bucket: `s3://bakalaurs/`
  - `processed/reformat_data/` = `2506` objekti, `44.94 GiB`
- AMS Spaces bucket: `s3://bakalaurs-ams/`
  - `processed/reformat_data/` = `2506` objekti, `44.94 GiB`
- Metadata AMS pusē:
  - `s3://bakalaurs-ams/meta/run_data_meta.csv`
  - `s3://bakalaurs-ams/meta/walk_data_meta.csv`
- JSON sanity check (20 failu izlase no Spaces):
  - parse errors = `0`
  - missing required keys = nav
  - mode mix: `run_only`, `walk_only`, `both` (sagaidāma struktūra)

### 18.2 Immediate next actions (izpildes secība)

1. **(Opcionāli, FRA CPU)** Validēt metodiku lokāli uz volume: repozitorijs + `ric_manifest.csv` / spliti + symlink uz `reformat_data`, tad `python scripts/train_anomaly.py --config config/anomaly_train_v1.yaml --epochs 1` iekš `tmux` (sk. §17.6). Tas **neaizstāj** GPU eksperimentu, bet dod reproducējamu “priekš-GPU” pierādījumu.
2. Izveidot GPU droplet `ams3` reģionā ar **AI/ML-ready** image.
3. Veikt GPU vides pārbaudi:
   - `nvidia-smi`
   - CUDA/PyTorch pieejamības tests.
4. Palaist nelielu subset smoke test:
   - datu ielāde no `s3://bakalaurs-ams/processed/reformat_data/`
   - 1–2 mini-batch treniņa soļi + checkpoint rakstīšana.
5. Palaist pilno treniņu:
   - regulāra checkpoint/log rakstīšana uz AMS Spaces.
6. Pēc treniņa:
   - rezultātu metrikas/ploti,
   - dropleta apturēšana/iznīcināšana.

### 18.3 Training readiness gate (go/no-go pirms pilnā GPU run)

Pilnais treniņš drīkst sākties tikai tad, ja:

1. `nvidia-smi` darbojas bez kļūdām.
2. `torch.cuda.is_available()` atgriež `True`.
3. Dataloader ielasa AMS datus bez schema/shape kļūdām.
4. Smoke test pabeidzas bez OOM/IO kļūdām.
5. Checkpoint fails tiek veiksmīgi ierakstīts `s3://bakalaurs-ams/checkpoints/`.
6. Treniņa konfigurācija (model/loss/lr/batch/split) ir nofiksēta.

Ja kaut viens punkts neizpildās, pilnais GPU treniņš netiek palaists.

### 18.4 Cost-control protocol (GPU izmaksu kontrole)

1. GPU droplet tiek palaists tikai tieši pirms treniņa.
2. Pirms pilnā run vienmēr izpildīt īsu smoke test.
3. Checkpoint jāglabā periodiski (piem., ik N epohām), lai neveidotu pilnu restartu.
4. Pēc run:
   - saglabāt final artefaktus Spaces,
   - apturēt/iznīcināt GPU droplet.
5. Nelietot GPU datu lejupielādei/ekstrakcijai, ja to var izdarīt ar CPU droplet.

### 18.5 Artifacts un outputs (mērķa struktūra Spaces)

- Dati:
  - `s3://bakalaurs-ams/processed/reformat_data/`
  - `s3://bakalaurs-ams/meta/`
- Treniņa artefakti:
  - `s3://bakalaurs-ams/checkpoints/`
  - `s3://bakalaurs-ams/logs/`
- Novērtēšanas rezultāti:
  - `s3://bakalaurs-ams/results/metrics/`
  - `s3://bakalaurs-ams/results/plots/`
  - `s3://bakalaurs-ams/results/tables/`

Obligātie gala artefakti:
- labākais model checkpoint,
- treniņa konfigurācijas fails,
- val/test metriku tabula,
- galvenie grafiki (loss/accuracy/F1 vai atbilstošās metrikas).

### 18.6 Priekš-GPU (CPU) treniņa artefakti lokālajā repozitorijā

Pēc veiksmīga `scripts/train_anomaly.py` palaišanas (sk. `docs/AMS3_PRE_GPU_RUNBOOK.md`) parasti rodas:

- `checkpoints/<run_id>/last.pt`, `best.pt` — checkpoint ar `model_state`, `input_size`, `hidden_size` u.c.;
- `logs/<run_id>/run_metadata.json`, `epoch_metrics.csv` — konfigurācijas hash, sēkla, git commit, epoku metrikas;
- `results/<run_id>/train_summary.json` — kopsavilkums.

Novērtēšanai: `scripts/evaluate_anomaly.py` raksta `metrics.json`, `thresholds.json`, `per_sample_scores.csv` izvēlētajā `--output-dir`. Šos failus var citēt bakalaura darbā kā **pirmās pilnas plūsmas** (dati → split → modelis → metrika) pierādījumu pirms mērogošanas uz GPU.

