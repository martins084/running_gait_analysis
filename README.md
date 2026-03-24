# Running Gait Analysis System

Automated running gait analysis using deep learning and computer vision.  
Built as a bachelor's thesis project — analyzes running form from regular video without expensive lab equipment.

## What It Does

- **Pose detection** — MediaPipe BlazePose extracts 33 body landmarks per frame  
- **Feature extraction** — joint angles, stride length, cadence, symmetry, vertical oscillation  
- **Temporal smoothing** — Savitzky–Golay filtering to eliminate pose jitter  
- **ML classification** — CNN-LSTM for gait phase detection (contact / swing / push-off)  
- **REST API** — FastAPI server: upload a video, get metrics back  

## Quick Start

```bash
# Clone and enter project
cd running_gait_analysis

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode (so imports work)
pip install -e .

# Run tests
pytest tests/ -v

# Start the API server
python api/app.py
# → http://localhost:8000/docs
```

## Project Structure

```
running_gait_analysis/
├── core/                   Core processing modules
│   ├── pose_detection.py       MediaPipe wrapper
│   ├── feature_extractor.py    Biomechanical feature math
│   ├── gait_analyzer.py        End-to-end pipeline
│   └── video_processor.py      Frame extraction
├── models/                 ML model definitions & weights
├── api/                    FastAPI server
│   └── app.py
├── utils/                  Visualization, helpers
│   └── visualization.py
├── tests/                  Unit & integration tests
├── data/                   Datasets (git-ignored)
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── notebooks/              Jupyter notebooks for exploration
├── requirements.txt
├── setup.py
└── README.md
```

## Tech Stack

| Layer            | Tool                          |
|------------------|-------------------------------|
| Pose detection   | MediaPipe BlazePose           |
| Deep learning    | PyTorch + Lightning           |
| Computer vision  | OpenCV                        |
| API              | FastAPI + Uvicorn             |
| Data processing  | NumPy, Pandas, SciPy          |
| Visualization    | Matplotlib, Plotly            |

## Accuracy Targets

| Metric                       | Target        |
|------------------------------|---------------|
| Pose detection (PCK)         | > 0.85        |
| Stride metrics error         | < 5%          |
| Ground contact accuracy      | > 90%         |
| Gait phase classifier        | > 85% val acc |
| Processing time (1-min video)| < 5 minutes   |

## License

Academic use — bachelor's thesis project.
