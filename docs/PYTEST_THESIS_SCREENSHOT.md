# Pytest izpilde — ekrānuzņēmums bakalaura / disertācijas darbam

Šis ceļvedis palīdz iegūt **skaidru, lasāmu** termināļa izvadi (`cmd` vai PowerShell), ko var ielikt darbā kā attēlu.

---

## 1. Priekšnosacījumi

1. Atver **Windows Terminal** vai **PowerShell** (modernāks fonts un krāsas nekā vecajam `cmd.exe`).
2. Palielini fontu pirms ekrānuzņēmuma: **Ctrl +** vai iestatījumos (piem. **Consolas / Cascadia Code**, 14–16 pt).
3. Pārliecinies, ka esi **projekta saknē** `running_gait_analysis` un ka **venv** ir aktivizēts.

---

## 2. Komandas (kopēt kā vienu bloku vai secīgi)

Izvēlies **vai nu garu sarakstu**, **vai kompaktu kopsavilkumu** — abi der disertācijai.

### 2a. Kompakts kopsavilkums (ieteicams, ja negrib garu sarakstu)

Rāda pytest sesijas galviņu, **punktiņus pa testu failiem** (viens `.` = viens tests) un beigās **`50 passed in X.XXs`**. Nav 50 atsevišķu rindu ar testu nosaukumiem.

**PowerShell / cmd** (pēc `cd` un `activate`):

```powershell
pytest tests -q --tb=no
```

- **`-q`** (*quiet*) — īsa izvade.  
- **`--tb=no`** — ja kāds tests kadreiz kritīs, bez gara “traceback” (attēlam tīrāk).

Vēl īsāka galvene (bez `platform`/`plugins` bloka), ja grib tikai būtisko:

```powershell
pytest tests -q --tb=no --no-header
```

Tad attēlā paliek galvenokārt: punktiņu rindas pa failiem + **`50 passed`**.

---

### 2b. Garš saraksts (katrs tests savā rindā)

```powershell
cd "C:\Users\varna\OneDrive\Desktop\BAKALURA DARBS\running_gait_analysis"
.\venv\Scripts\Activate.ps1
pytest tests -v --tb=short
```

**Nemēslo `-q` kopā ar `-v`:** `-q` saīsina izvadi; ar **tikai** `-v` katrs tests ir atsevišķa rinda, piem.  
`tests/test_api_app.py::test_analyze_rejects_unsupported_extension PASSED`.

**Command Prompt (`cmd.exe`):**

```cmd
cd /d "C:\Users\varna\OneDrive\Desktop\BAKALURA DARBS\running_gait_analysis"
venv\Scripts\activate.bat
pytest tests -v --tb=short
```

### Karodziņu tabula

| Karodziņa | Kāpēc |
|-----------|--------|
| `tests` | Skaidri redzams, ka skenēta `tests/` mape. |
| `-v` | Katrs tests atsevišķā rindiņā. |
| `-q` | Kompakti: punkti + kopējais `passed`. |
| `--tb=short` / `--tb=no` | Īss vai nav traceback kļūdām. |
| `--no-header` | Mazāk sesijas metadatu kompaktam attēlam. |

---

## 3. Tikai “cik testu savākts” (bez izpildes)

Ja vajag miniatūru rindiņu, ka kopā ir 50 testi, **bez** to palaišanas:

```powershell
pytest tests --collect-only -q
```

Parasti joprojām izdrukā visus nosaukumus; **īsākais** palaides kopsavilkums paliek **§2a** (`pytest tests -q --tb=no`).

---

## 4. Ja gribi **bez krāsām** (dažiem Word/PDF labāk izskatās pelēks teksts)

```powershell
$env:PYTEST_THEME = "none"
pytest tests -q --tb=no
```

(Vai ar `-v` — tāpat pirms tam `PYTEST_THEME=none`.)

---

## 5. Ekrānuzņēmuma saturs, ko lasītājs saprot

**Kompaktam režīmam (`-q`):** `cd`, `activate`, komanda, īss pytest ievads (ja nav `--no-header`), punktiņu rindas pa failiem, **`50 passed in X.XXs`**.

**Verbose (`-v`):** papildus garš saraksts `file.py::test_name PASSED`.

---

## 6. Īss paraksts zem attēla darbā (angļu vai latviski)

*Piemērs (EN), kompakti:*  
“Summary of the automated test suite (`pytest -q`): all 50 tests in `tests/` passed.”

*Piemērs (LV):*  
“Automātisko testu kopsavilkums (`pytest -q`): visi 50 testi mapē `tests/` iziet veiksmīgi.”

*Ja izmantots `-v`:* pieminēt “verbose mode” / “detalizēts izvades režīms”.

---

## 7. Ja `Activate.ps1` neļauj palaist (ExecutionPolicy)

Vienreiz PowerShell kā administrators:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Vai aktivē venv tikai ar `cmd` un `activate.bat`, kā augšā.

---

*Pielāgo `cd` ceļu, ja projekts ir citā diskā vai mapē.*
