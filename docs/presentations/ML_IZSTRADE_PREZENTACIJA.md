# ML izstrāde — vizualizācijas prezentācijai

Šajā failā ir **Mermaid** diagrammas, ko vari ielikt PowerPoint / Google Slides / Keynote.

## Kā iegūt attēlu

1. Atver [https://mermaid.live](https://mermaid.live)
2. Iekopē zemāk esošo ` ```mermaid ... ``` ` bloku (bez ārējām atzīmēm — tikai saturu starp mermaid tagiem)
3. **Export** → PNG vai SVG
4. Ievieto slaidā

Ieteikums: eksportē **SVG** ja vajag maksimālu asumu projektorā.

---

## 1. attēls — ML izstrādes cikls (projekta kontekstā)

Šis parāda tipisko **MLOps / izstrādes** plūsmu: dati → iezīmes → modelis → novērtēšana → ieviešana.

```mermaid
flowchart LR
  subgraph data [Dati un sagatavošana]
    V[Video ieraksti]
    P[MediaPipe BlazePose]
    F[Secību iezīmes 66D]
  end

  subgraph labels [Marķēšana]
    L[Fāžu etiķetes: stance / swing / push]
  end

  subgraph train [Apmācība]
    T[CNN-LSTM PyTorch]
    M[Modeļa versija .pt]
  end

  subgraph eval [Novērtēšana]
    E[Validācijas metrika: accuracy / F1]
    X[Salīdzinājums ar bāzes līniju]
  end

  subgraph deploy [Ieviešana]
    A[FastAPI / batch]
    U[Atjaunināšana bez dīkstāves]
  end

  V --> P --> F
  F --> L
  F --> T
  L --> T
  T --> M
  M --> E --> X
  M --> A
  A --> U
```

---

## 2. attēls — `GaitPhaseClassifier` datu plūsma (arhitektūra)

Atbilst `models/gait_classifier.py`: ieeja `(batch, seq, 66)` → Conv1d → LSTM → logits.

```mermaid
flowchart TB
  IN["Ieeja: pose secība<br/>batch × seq_len × 66<br/>33 locītavas × x,y"]

  subgraph conv [Laika konvolūcija]
    C1["Conv1d 66→64, k=3"]
    C2["Conv1d 64→128, k=3"]
  end

  subgraph seq [Secību modelis]
    L["LSTM hidden=128"]
    FC["Linear → 64 → num_phases"]
    OUT["Izvade: logits + LSTM stāvoklis"]
  end

  IN --> C1 --> C2 --> L --> FC --> OUT
```

---

## 3. attēls — pilna sistēma: CV + ML + API (augsta līmeņa)

Noderīgs kā **1. slaida** pārskats.

```mermaid
flowchart TB
  subgraph client [Klients]
    UI[Web UI vai API klients]
  end

  subgraph api [Backend]
    FA[FastAPI]
    PD[PoseDetector]
    FE[FeatureExtractor]
    GC[GaitPhaseClassifier — kad pievienots]
  end

  subgraph store [Glabāšana]
    DB[(SQLite / rezultāti)]
    FS[Video / JSON artefakti]
  end

  UI -->|POST /analyze| FA
  FA --> PD
  PD --> FE
  FE --> GC
  GC --> DB
  FE --> DB
  PD --> FS
```

---

## Īss teksts slaidam (bulleti)

- **Ieeja**: normalizētas pozu koordinātas secībā (66 kanāli uz laika soli).
- **Modelis**: CNN pār laika asi + LSTM kontekstam starp kadriem.
- **Uzdevums**: gaitas fāžu klasifikācija (vai anomāliju detekcija ar autoencoder).
- **Izstrāde**: PyTorch → saglabāts `.pt` → integrācija API vai atsevišķa inferenci.

---

## Licences un avoti slaidam

- MediaPipe BlazePose — Google MediaPipe dokumentācija.
- PyTorch — pytorch.org.
