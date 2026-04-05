# Python automatizēto testu detalizēts apraksts (`tests/`)

Šis dokuments apraksta **katru** `pytest` testu projektā *running_gait_analysis*: mērķi, testa datu veidošanu, pārbaudītos apgalvojumus (`assert`) un saistību ar produkcijas kodu. Kopā **50** testu funkcijas.

**Rīks:** `pytest` (ar `tmp_path`, `monkeypatch`, `TestClient`).

**Svarīgi:** API testi **nemēģina** palaist īstu MediaPipe vai OpenCV kodēšanu — `PoseDetector`, `FeatureExtractor` un `create_annotated_video` tiek **aizvietoti** ar vieglām mock klasēm, lai testi būtu ātri un deterministiski.

---

## 1. `test_api_app.py` — FastAPI (`api/app.py`)

**Moduļa mērķis:** pārbaudīt HTTP atbildes kodus, `detail` ziņojumus, SQLite ierakstus un failu piegādi, nepalaižot smago video apstrādi.

### Palīgklases (mock)

| Klase / funkcija | Uzvedība |
|------------------|----------|
| `_OkDetector` | `process_video` ieraksta minimālu derīgu `poses.json` (fps 30, viens kadrs, 33 punkti). |
| `_FailDetector` | `process_video` met `RuntimeError("bad/corrupt stream")`. |
| `_SlowDetector` | `process_video` gul 0.2 s (simulē ilgu darbu). |
| `_OkExtractor` | `extract_all_features` atgriež `{"stride_metrics": {"cadence_steps_per_min": 170.0}}`. |
| `_fake_annotator` | `create_annotated_video` ieraksta baitus `fake-mp4` failā. |
| `_client_with_tmpdirs` | `UPLOAD_DIR` un `RESULTS_DIR` novirza uz `tmp_path`, izveido mapes, atgriež `TestClient(app)`. |

---

### `test_analyze_rejects_unsupported_extension`

- **Mērķis:** neatļauts augšupielādes paplašinājums netiek apstrādāts kā video.
- **Darbība:** POST `/analyze` ar lauku `video`: fails `run.txt`, MIME `text/plain`, saturs `not-a-video`.
- **Pārbaudes:** `status_code == 400`; JSON `detail` satur tekstu `"Unsupported format"`.
- **Saistība:** `ALLOWED_EXTENSIONS` validācija pirms detektora.

---

### `test_analyze_rejects_empty_upload`

- **Mērķis:** tukšs fails netiek pieņemts.
- **Darbība:** POST ar `run.mp4`, tukšs `b""`, `video/mp4`.
- **Pārbaudes:** `400`; `detail` (bez reģistra atšķirības) satur `"empty"`.
- **Saistība:** servera pārbaude uz tukšu augšupielādi.

---

### `test_analyze_maps_pose_failure_to_422`

- **Mērķis:** pozas analīzes kļūda tiek kartēta uz 422 Unprocessable Entity.
- **Darbība:** `_FailDetector`; POST ar derīgu MP4 imitāciju.
- **Pārbaudes:** `422`; `detail` satur `"Pose analysis failed"`.
- **Saistība:** kļūdu apstrāde ap `process_video`.

---

### `test_analyze_timeout_returns_504`

- **Mērķis:** pārsniegts analīzes laika limits.
- **Darbība:** `_SlowDetector`; `ANALYZE_TIMEOUT_SEC` uzlikts uz `0.01`; POST ar MP4.
- **Pārbaudes:** `504`; `detail` satur `"timed out"` (reģistrneatkarīgi).
- **Saistība:** `asyncio.wait_for` vai līdzīgs timeout ap analīzi.

---

### `test_analyze_persists_result_in_sqlite`

- **Mērķis:** veiksmīga analīze ieraksta rindu datubāzē.
- **Darbība:** `_OkDetector`, `_OkExtractor`, `_fake_annotator`; POST `/analyze`.
- **Pārbaudes:** atbilde `200`; `analysis_id` no JSON; fails `_db_path()` eksistē; SQL `SELECT id, status FROM analyses WHERE id = ?` → rinda ar `status == "completed"`.
- **Saistība:** SQLite shēma un `_save_analysis` loģika.

---

### `test_get_results_reads_from_sqlite_when_json_missing`

- **Mērķis:** ja `*_result.json` pazudis, rezultāts joprojām jānolasa no DB.
- **Darbība:** veiksmīga POST; dzēš `RESULTS_DIR / f"{analysis_id}_result.json"`; GET `/results/{analysis_id}`.
- **Pārbaudes:** `200`; JSON `id` sakrīt ar `analysis_id`.
- **Saistība:** fallback no SQLite, kad fails nav uz diska.

---

### `test_health_endpoint_returns_ok`

- **Mērķis:** dzīvības pārbaude.
- **Darbība:** GET `/health`.
- **Pārbaudes:** `200`; ķermenis tieši `{"status": "ok"}`.

---

### `test_get_results_missing_id_returns_404`

- **Mērķis:** neeksistējošs analīzes ID.
- **Darbība:** GET `/results/does-not-exist`.
- **Pārbaudes:** `404`; `detail` satur `"not found"`.

---

### `test_download_missing_video_returns_404`

- **Mērķis:** lejupielāde neeksistējošam ID.
- **Darbība:** GET `/download/does-not-exist`.
- **Pārbaudes:** `404`; `detail` satur `"not found"`.

---

### `test_get_poses_returns_stored_json`

- **Mērķis:** GET `/poses/{id}` atgriež saglabāto BlazePose JSON.
- **Darbība:** manuāli ieraksta `results/pose-test-id_poses.json` ar `{"fps": 30, "poses": []}`; GET `/poses/pose-test-id`.
- **Pārbaudes:** `200`; `Content-Type` sākas ar `application/json`; parsētā `fps == 30`.

---

### `test_get_poses_missing_returns_404`

- **Mērķis:** nav poses faila.
- **Darbība:** GET `/poses/does-not-exist`.
- **Pārbaudes:** `404`.

---

### `test_download_returns_mp4_for_completed_analysis`

- **Mērķis:** pēc veiksmīgas analīzes lejupielāde ir video.
- **Darbība:** pilna veiksmīga POST; GET `/download/{analysis_id}`.
- **Pārbaudes:** `200`; `Content-Type` sākas ar `video/mp4`.
- **Saistība:** `FileResponse` ar annotēto MP4.

---

## 2. `test_feature_extraction.py` — `core/feature_extractor.py`

**Moduļa mērķis:** matemātika locītavu leņķiem, simetrijas indeksam, soļu metrikām un `_smooth_sequence` (Savitzky–Golay).

**Fixture:** `extractor` → `FeatureExtractor(fps=30)`.

---

### `TestJointAngles.test_right_angle`

- **Mērķis:** `_angle_at` ar taisnu leņķi.
- **Ievade:** trīs punkti 2D (x,y no `p1`–`vertex`–`p3`), formē ~90°.
- **Pārbaudes:** `abs(angle - 90) < 1` grādi.
- **Saistība:** kosinusu likums `FeatureExtractor._angle_at`.

---

### `TestJointAngles.test_straight_line`

- **Mērķis:** kolineāri punkti → 180°.
- **Pārbaudes:** `abs(angle - 180) < 1`.

---

### `TestJointAngles.test_acute_angle`

- **Mērķis:** zināma smaila leņķa ģeometrija.
- **Pārbaudes:** `abs(angle - 45) < 2`.

---

### `TestSymmetry.test_perfect_symmetry`

- **Mērķis:** simetriska kreisā/labā ķēde dod augstu `compute_symmetry`.
- **Ievade:** `_make_landmarks()` — gurni ap 0.5, ceļi un potītes spoguļsimetrijā.
- **Pārbaudes:** `sym > 0.90`.

---

### `TestSymmetry.test_asymmetric`

- **Mērķis:** nobīdīts kreisais ceļš samazina simetriju.
- **Ievade:** `left_knee` uz `[0.1, 0.7, ...]`.
- **Pārbaudes:** `sym < 0.80`.

---

### `TestStrideMetrics.test_synthetic_stride`

- **Mērķis:** `compute_stride_metrics` atrod soļus no sinusoidālas kreisās potītes kustības.
- **Ievade:** 4 s × 30 fps kadrus; kreisā potīte (indekss 27) ar `sin` pa fāzi 1.5 Hz.
- **Pārbaudes:** `metrics` nav tukšs; `num_strides_detected >= 2`; ir `cadence_steps_per_min` un tā > 50.
- **Saistība:** `find_peaks` uz potītes Y un kadences formula.

---

### `TestTemporalSmoothing.test_short_sequence_is_returned_unchanged`

- **Mērķis:** ja kadru skaits < `smooth_window` (7), gludināšana netiek lietota.
- **Ievade:** 5 kadrus ar mainīgu potītes x.
- **Pārbaudes:** garums saglabāts; katrs izvades kadrs `allclose` ar ievadi.
- **Saistība:** `_smooth_sequence` agrīnā iziešana.

---

### `TestTemporalSmoothing.test_none_frames_are_preserved_in_place`

- **Mērķis:** trūkstošās detekcijas (`None`) netiek “iedomātas” gludināšanā.
- **Ievade:** secība ar `None` pozīcijās 1 un 4; `smooth_window=5`.
- **Pārbaudes:** `out[1]` un `out[4]` ir `None`; pārējie ir `np.ndarray`.

---

### `TestTemporalSmoothing.test_smoothing_reduces_high_frequency_noise`

- **Mērķis:** SG samazina MSE pret gludu referenci salīdzinājumā ar trokšņainu signālu.
- **Ievade:** 25 kadri; `clean` sinusoida; `noisy = clean +` alternējošs mazs troksnis; gludināts tikai potītes x.
- **Pārbaudes:** `mse(smoothed, clean) < mse(noisy, clean)`.

---

## 3. `test_visualization.py` — `utils/visualization.py`

**Palīgfunkcija:** `_landmarks_all_visible()` — 33×4 stāvoša poza, pilna redzamība, punkti izkārtoti COM aprēķinam.

---

### `test_draw_pose_adds_green_and_bone_pixels`

- **Mērķis:** zīmēšana maina kadru; kauli/augstas redzamības locītavas izmanto zaļo kanālu (BGR).
- **Darbība:** tukšs 640×480 kadrs; `draw_pose`; `cv2.absdiff` summa > 1000; zaļā kanāla `max > 80`.

---

### `test_draw_pose_low_visibility_uses_red_joint`

- **Mērķis:** zema `visibility` → sarkanās locītavas krāsa.
- **Darbība:** `lm[16, 3] = 0.2` (labais plaukstas locītava).
- **Pārbaudes:** sarkanā kanāla (`[:,:,2]`) maksimums > 100.

---

### `test_skeleton_connections_cover_expected_edges`

- **Mērķis:** skeleta grafs derīgs BlazePose indeksiem.
- **Pārbaudes:** katram `(i,j)` no `SKELETON_CONNECTIONS`: `0 <= i,j < 33`.

---

### `test_visibility_strictly_above_half_is_green_joint`

- **Mērķis:** `visibility > 0.5` → zaļš punkts.
- **Darbība:** tikai deguns (0) centrā ar `0.51`; pārējie punkti ar zemu redzamību ārpus redzamās zonas.
- **Pārbaudes:** pikseļa (320,240): `g > r + 50`.

---

### `test_visibility_at_or_below_half_is_red_joint`

- **Mērķis:** `visibility == 0.5` krīt zem sliekšņa (`> 0.5` ir `False`).
- **Pārbaudes:** `r >= g` centrālajā pikselī.

---

### `test_draw_pose_accepts_list_of_lists_for_landmarks`

- **Mērķis:** JSON-kompatibls masīvs nedod izņēmumu.
- **Darbība:** `lm.tolist()`; mazs kadrs 100×100.
- **Pārbaudes:** `frame.sum() > 0`.

---

### `test_compute_segment_weighted_com_symmetric_pose_near_midline`

- **Mērķis:** `compute_segment_weighted_com_xy` simetriskai pozai dod COM x tuvu 0.5.
- **Pārbaudes:** `com` nav `None`; `abs(com[0] - 0.5) < 0.12`.

---

### `test_draw_pose_returns_com_coordinates`

- **Mērķis:** `draw_pose` atgriež COM kā `(x_px, y_px)` vai `None`.
- **Pārbaudes:** atgrieztā vērtība nav `None`; `len(out) == 2`.

---

## 4. `test_pose_foot_sanitize.py` — `utils/pose_foot_sanitize.py`

---

### `test_extreme_lateral_ankle_is_clamped_toward_knee`

- **Mērķis:** nefizioloģiski tālu kreisā potīte (“karogs”) tiek pavilkta pie ceļa.
- **Ievade:** 8 kadrus ar sintētisku pozu — potīte x=0.02, ceļs ~0.50; pirms: attālums ceļš–potīte > 0.35.
- **Darbība:** `sanitize_pose_sequence(poses)` in-place.
- **Pārbaudes:** pēc apstrādes apakšstilba garums (2D) `<= 0.23` (atbilst kodā ~0.22 cap ar mazu toleranci).

---

### `test_ankle_identity_continuity_avoids_left_right_swap`

- **Mērķis:** kreisās/labās potītes apmaiņa starp kadriem tiek labota pēc kustības turpinājuma.
- **Ievade:** kadrs 0 normāls; kadrs 1 apmainītas potīšu pozīcijas.
- **Pārbaudes:** pēc sanitizācijas kreisās potītes x < labās potītes x (saskanība ar kadru 0).

---

## 5. `test_pose_detection.py` — `core/pose_detection.py`

**Fixture:** viens `PoseDetector` uz visu moduli (`scope="module"`) — ielādē MediaPipe modeli vienu reizi.

---

### `test_landmark_names_count`

- **Mērķis:** BlazePose API — 33 anatomiskie nosaukumi.
- **Pārbaudes:** `len(LANDMARK_NAMES) == 33`.

---

### `test_detect_on_blank_image`

- **Mērķis:** tukšā bilde nesatur personu.
- **Darbība:** melns 640×480 `uint8`; `detect_pose`.
- **Pārbaudes:** `landmarks is None` un `raw is None`.

---

### `test_detect_returns_correct_shape`

- **Mērķis:** līgums par izvades formu, ja detekcija būtu veiksmīga.
- **Darbība:** tukša bilde — parasti `None`; ja kādreiz būtu landmarks, forma `(33, 4)`.
- **Piezīme:** integrācijas tests ar īstu cilvēka foto nav šeit.

---

## 6. `test_video_processor.py` — `core/video_processor.py`

**Palīgfunkcija:** `_write_short_mp4` — OpenCV `mp4v` īss sintētisks klips.

---

### `test_get_video_info_frame_count_and_fps`

- **Mērķis:** `get_video_info` atgriež pareizus metadatus.
- **Pārbaudes:** `frame_count == 20`, fps ≈ 30, izmēri 320×240, `duration_sec > 0`.

---

### `test_extract_frames_respects_target_fps`

- **Mērķis:** `fps_target=10` uz 60 kadriem @ 30 fps → aptuveni trešdaļa saglabāto kadru.
- **Pārbaudes:** saglabāto skaits diapazonā 15–25; JPG skaits = `n`.

---

### `test_normalize_resolution_downscales_wide_frame`

- **Mērķis:** ļoti plats kadrs tiek samazināts līdz `MAX_WIDTH`.
- **Pārbaudes:** izvades platums == `VideoProcessor.MAX_WIDTH`; augstums < 720.

---

### `test_cannot_open_missing_file`

- **Mērķis:** neeksistējošs ceļš → `IOError`.
- **Darbība:** `pytest.raises(IOError)` ap `get_video_info` uz neesošu ceļu.

---

### `test_sub_short_video_under_one_second`

- **Mērķis:** īss klips (< 1 s) joprojām derīgs.
- **Pārbaudes:** `frame_count == 10`, `duration < 1 s`, `extract_frames` atgriež 10.

---

### `test_narrow_frame_not_resized`

- **Mērķis:** šaurāki par `MAX_WIDTH` netiek mērogoti.
- **Pārbaudes:** `_normalize_resolution` saglabā `shape`.

---

## 7. `test_evaluate_ground_contact.py` — `scripts/evaluate_ground_contact.py`

**Palīgfunkcijas:** `_load_eval_mod()` dinamiski ielādē skriptu; `_poses_with_peaks()` — sinusoidāla kreisās potītes y.

---

### `test_ankle_ground_score_series_runs`

- **Mērķis:** `ankle_ground_score_series` atgriež masīvu garumā kā poses skaits, bez NaN/inf.
- **Pārbaudes:** `len(s) == len(poses)`; `np.isfinite(s).all()`.

---

### `test_run_evaluation_writes_metrics`

- **Mērķis:** pilna novērtēšanas plūsma ar etiķēm un izvades failiem.
- **Darbība:** poses JSON; CSV ar `frame_id`, `is_contact`; `run_evaluation`.
- **Pārbaudes:** eksistē `ground_contact_metrics.csv`, `ground_contact_summary.txt`; atgrieztajā `res` ir F1 jēdziena līmenis; `sensitivity`, `specificity` ∈ [0,1].

---

## 8. `test_batch_feature_export.py` — `scripts/batch_feature_export.py`

---

### `test_batch_export_creates_three_outputs`

- **Mērķis:** `export_one` no `*_poses.json` ģenerē feature JSON, kadru CSV un ierakstu kopsavilkumā.
- **Darbība:** 120 kadru sintētiska poza; `monkeypatch.chdir(ROOT)`; `export_one` ar `FeatureExtractor(fps=30)`, `force=True`.
- **Pārbaudes:** `abc123_features.json` un `abc123_features_frames.csv` eksistē; JSON satur `features.stride_metrics`; CSV 121 rinda (1 header + 120); summary satur `abc123` un `synthetic_test.mp4`.

---

## 9. `test_validate_features.py` — `scripts/validate_features.py`

---

### `test_validate_features_known_offset`

- **Mērķis:** salīdzinājums starp manuāli ievadītu un aprēķinātu leņķi.
- **Ievade:** features CSV: `left_knee_deg = 90`; manual CSV: `manual_angle = 88`, `angle_name = left_knee`; 5 rindas.
- **Pārbaudes:** `run_validation` iziešanas kods `0`; `angle_validation_metrics.csv` satur `left_knee` un `PASS` vai `2.0` (2° kļūda).

---

## 10. `test_scripts_edge_cases.py` — vairāki skripti

---

### `test_load_labels_csv_skips_bad_rows`

- **Mērķis:** bojāta CSV rinda (`oops`) tiek izlaista.
- **Pārbaudes:** paliek tikai derīgie kadri 0 un 2; `len == 2`.

---

### `test_load_labels_csv_empty_after_bad_rows_raises`

- **Mērķis:** ja visas rindas nederīgas → skaidra kļūda.
- **Pārbaudes:** `ValueError` ar tekstu `No valid label rows`.

---

### `test_run_evaluation_rejects_empty_poses`

- **Mērķis:** tukšs `poses` masīvs.
- **Pārbaudes:** `ValueError` ar `zero frames`.

---

### `test_validate_features_no_matches_returns_two`

- **Mērķis:** ja `video_name` manuālajā un feature CSV neatkrīt, nav salīdzinājumu.
- **Pārbaudes:** `run_validation` atgriež iziešanas kodu `2`.

---

### `test_batch_export_rejects_invalid_poses_json`

- **Mērķis:** JSON bez `poses` lauka.
- **Pārbaudes:** `ValueError` ar `poses` pattern.

---

### `test_validate_features_rejects_headerless_csv`

- **Mērķis:** tukšs fails nav derīgs manuālais CSV.
- **Pārbaudes:** `ValueError` ar `header`.

---

## 11. Kopsavilkums tabulā

| Fails | Testu skaits | Galvenā saistība |
|-------|-------------|------------------|
| `test_api_app.py` | 12 | FastAPI, SQLite, kļūdas kodi |
| `test_feature_extraction.py` | 9 | Leņķi, simetrija, soļi, SG |
| `test_visualization.py` | 8 | Skelets, COM, redzamība |
| `test_video_processor.py` | 6 | Video metadati, kadri |
| `test_scripts_edge_cases.py` | 6 | CSV/JSON robežas |
| `test_pose_detection.py` | 3 | PoseDetector smoke |
| `test_evaluate_ground_contact.py` | 2 | Zemes kontakts |
| `test_pose_foot_sanitize.py` | 2 | Kāju sanitizācija |
| `test_batch_feature_export.py` | 1 | Batch eksports |
| `test_validate_features.py` | 1 | Leņķu validācija |
| **Kopā** | **50** | |

---

*Dokuments atbilst `tests/` saturam repozitorijā. Ja testi tiek pievienoti, atjaunināt šo failu un `pytest` kopsummu.*
