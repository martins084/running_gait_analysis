# Skrējiena gaitas analīze — programmatūras arhitektūras blokshēmas

Šis dokuments apkopo **plānoto un pašreiz implementēto** sistēmas arhitektūru. Diagrammas renderējas GitHub/GitLab un VS Code/Cursor (Mermaid spraudnis). Drukāšanai: eksportēt PNG no [mermaid.live](https://mermaid.live).

---

## 1. Augsta līmeņa skats (lietotājs → serveris → dati)

Plānotā pilnā struktūra: viena tīmekļa lietotne, REST API, modulāra apstrāde, failu un SQLite glabāšana.

```mermaid
flowchart TB
  subgraph client["Klienta līmenis"]
    U[Lietotājs / pārlūks]
    SPA["SPA: React + Vite\n(augšupielāde, rezultāti, grafiki, video)"]
    U --> SPA
  end

  subgraph server["Servera līmenis — FastAPI"]
    API["REST API\n/analyze /results /download /poses /health"]
    CORS["CORS / proxy\n(Vite dev)"]
    SPA -->|HTTPS fetch multipart| API
  end

  subgraph core["Apstrādes kodols — Python"]
    PD["PoseDetector\nMediaPipe BlazePose\nVIDEO režīms"]
    SAN["sanitize_pose_sequence\n(kāju ģeometrija)"]
    FE["FeatureExtractor\nSavitzky–Golay, metrikas\nCOM, leņķi, soļi"]
    VIS["create_annotated_video\n(skeleta MP4)"]
    PD --> SAN --> FE
    SAN --> VIS
    FE --> OUT["features dict\n(JSON atbilde)"]
  end

  subgraph ml["Plānotā paplašinājuma slānis (daļēji definēts kodā)"]
    GC["GaitPhaseClassifier\n1D-CNN + LSTM"]
    AD["GaitAnomalyDetector"]
    FE -.->|nākotnē: fāžu rinda| GC
    FE -.->|nākotnē| AD
  end

  subgraph storage["Glabāšana"]
    SQL[(SQLite\nanalysis_results.db)]
    FS["Failu sistēma\nuploads/, results/\nposes.json, annotated.mp4"]
    API --> SQL
    API --> FS
  end

  API --> PD
  FE --> API
  VIS --> FS
  OUT --> API
```

**Leģenda:** cietā bultiņa — pašreizējā datu plūsma; pārtraukta līnija (`-.->`) — **plānota** ML saite, kad modelis ir apmācīts un pieslēgts API.

---

## 2. HTTP pieprasījumu un artefaktu plūsma

```mermaid
flowchart LR
  subgraph http["Galapunkti"]
    H["GET /health"]
    A["POST /analyze\nvideo → id"]
    R["GET /results/{id}\nJSON ar features"]
    D["GET /download/{id}\nannotated MP4"]
    P["GET /poses/{id}\npilnais BlazePose JSON"]
  end

  A --> R
  A --> D
  R --> P
```

---

## 3. Video apstrādes secība (kodola pipeline)

Tas pats, kas `process_video` / API iekšējā loģika: secīgi un skaidri moduļi.

```mermaid
flowchart LR
  V[Video MP4] --> PD[PoseDetector]
  PD --> J[poses.json\n33×4 katrā kadrā]
  J --> SAN[sanitize]
  SAN --> FE[FeatureExtractor\nsmoothing + metrikas]
  SAN --> VIS[annotated MP4\nskelets]
  FE --> F[features]
  F --> API[API atbilde / SQLite]
  VIS --> API
```

---

## 4. Frontend iekšējā loģika (plānotā UI arhitektūra)

```mermaid
flowchart TB
  subgraph fe["frontend/ — React"]
    R1["/ — AnalyzePage\nUploadZone"]
    R2["/results/:id — ResultsPage"]
    VP["VideoPanel\nvideo + canvas COM"]
    CH["Recharts\nleņķi, simetrija, COM"]
    SYNC["useVideoFrameSync\nlaiks ↔ kadra indekss"]
    R1 -->|POST analyze| R2
    R2 --> VP
    R2 --> CH
    VP --> SYNC
    CH --> SYNC
  end
```

---

## 5. Moduļu un mapes atbilstība (īss indekss)

| Bloks diagrammā | Repozitorija ceļš |
|-----------------|-------------------|
| PoseDetector | `core/pose_detection.py` |
| sanitize | `utils/pose_foot_sanitize.py` |
| FeatureExtractor | `core/feature_extractor.py` |
| create_annotated_video | `utils/visualization.py` |
| FastAPI | `api/app.py` |
| ML definīcijas | `models/gait_classifier.py` |
| SPA | `frontend/src/` |

---

## 6. Piezīmes disertācijai / darbam

- **Plānotā** arhitektūra ietver **pilnu cilpu**: video ievade → poza → sanitizācija → biomehāniskie lielumi → vizualizācija → JSON/DB → tīmekļa atskaite. **ML fāžu klasifikators** ir atzīmēts kā paplašinājums, jo arhitektūra (`GaitPhaseClassifier`) ir sagatavota, bet **produkcijas API** var vēl neeksportēt `phases_per_frame` visos scenārijos.
- Ja nepieciešams **viena lapa** ar vienu attēlu, izdrukājiet tikai **§1** diagrammu vai apvienojiet to ar institūcijas veidni.

---

*Atjaunināšanas avots: `docs/SYSTEM_OVERVIEW_AND_CURRENT_STATE.md`, `api/app.py`, `frontend/` struktūra.*
