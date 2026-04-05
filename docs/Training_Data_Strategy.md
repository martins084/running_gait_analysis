# Training Data Strategy

**[PLACEHOLDER — thesis content]** Replace the tables below with your finalized data plan.

---

## [PLACEHOLDER] Label source

| Topic | Your decision |
|-------|----------------|
| Primary label source | Hybrid: Heuristic (primary) + Manual validation sample|
| Ground truth for angles | Visual measurement from annotated frames (protractor-equivalent in video editing tool or manual degree estimation) |
| Ground contact | Automatic detection via ankle-y peak detection; validated against 1-2 manually labeled clips |

## [PLACEHOLDER] Splits & scale

| Split | Rule (video- / subject- / session-level) | Approx. N |
|-------|------------------------------------------|------------|
| Train | First 7 of 10 clips (by video, chronological order) | ~350-400 frames |
| Val | 2 middle clips (manually labeled + heuristic) | ~200-250 frames |
| Test | Last 1 clip (held-out, different condition if possible) | ~100-150 frames |

## [PLACEHOLDER] Augmentation & exclusions

- **Allowed augmentations:** _resolution / color / temporal crop — note constraints for running pose._
- **Exclusions:** _occlusion, multi-person, unstable tracking — link to `MediaPipe_Accuracy_Notes.md` if relevant._

TechniqueAllowed?Rationale for Running PoseTemporal cropping✅ YESCrop to 1-2 full gait cycles (e.g., frames 100-200). Acceptable for learning.Temporal speed variation✅ YES (±10%)Speed up/slow down clip by 0.9–1.1×. Realistic (treadmill variations). Adjust FPS in pose JSON.Spatial resolution change⚠️ LIMITEDDownsample to 720p (from 1080p). Avoid upsampling past original. Running pose robust to resolution if not too low (<480p).Brightness/contrast✅ YES±20% brightness, ±10% contrast. MediaPipe robust to lighting variations. Mirrors real-world outdoor running.Horizontal flip✅ YESMirror video left-right. Common in pose estimation. Doubles effective dataset size.Rotation❌ NOLarge rotation (>10°) breaks hip/knee joint angle definitions. Stick to near side-view camera.Occlusion simulation❌ NOAdding synthetic occlusion (e.g., masking arms) not needed; real occlusion already present in source videos.


## [PLACEHOLDER] Ethics & sources

- **External video rights / licenses:** _cite_
- **Human subjects:** _if applicable_

All 10 sample videos:

✅ License: Creative Commons (CC-BY, CC0) OR public domain
✅ Source: YouTube (CC filter), Pexels.com, Pixabay.com
✅ Commercial use: Not re-sold; thesis research + academic use only
✅ Attribution: Creator name + license noted in VIDEO_SOURCES.txt per video