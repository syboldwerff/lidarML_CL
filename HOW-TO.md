# HOW-TO: LiDAR Road Modeling Pipeline

Deze gids beschrijft alle commando's die nodig zijn om de LiDAR Road Modeling Pipeline te gebruiken, in de juiste volgorde.

---

## 📋 Inhoudsopgave

1. [Setup & Installatie](#1-setup--installatie)
2. [Demo Preprocessing Pipeline](#2-demo-preprocessing-pipeline)
3. [Preprocessing van Eigen Data](#3-preprocessing-van-eigen-data)
4. [QA Visualisatie](#4-qa-visualisatie)
5. [ML Training - BEV Segmentatie](#5-ml-training---bev-segmentatie)
6. [ML Training - Camera Detectie](#6-ml-training---camera-detectie)
7. [Inference & Voorspelling](#7-inference--voorspelling)
8. [Object Extractie](#8-object-extractie)
9. [Wegmodel Generatie](#9-wegmodel-generatie)
10. [Tests Uitvoeren](#10-tests-uitvoeren)

---

## 1. Setup & Installatie

### 1.1 Dependencies installeren (zonder Docker)

```bash
pip install -r requirements.txt
```

### 1.2 Docker omgeving bouwen - Preprocessing

```bash
docker build -f docker/Dockerfile.preprocess -t lidarml-preprocess .
```

### 1.3 Docker omgeving bouwen - Training/Inference

```bash
docker build -f docker/Dockerfile.train -t lidarml-train .
```

### 1.4 Complete omgeving starten met docker-compose

```bash
docker-compose up -d
```

### 1.5 Omgeving stoppen

```bash
docker-compose down
```

---

## 2. Demo Preprocessing Pipeline

### 2.1 Demo uitvoeren met synthetische data

```bash
python examples/demo_preprocessing_pipeline.py
```

Dit genereert:
- Output in `output/demo_preprocessing/`
- Metadata: `output/demo_preprocessing/meta_scan.json`
- Tiles: `output/demo_preprocessing/pre_tiles/`
- Goedgekeurde tiles: `output/demo_preprocessing/tiles_approved/`
- Review tiles: `output/demo_preprocessing/qa_review/`

---

## 3. Preprocessing van Eigen Data

### 3.1 Configuratie aanpassen

Bewerk `configs/preprocessing.yaml` naar wens:

```bash
# Voorbeeld: tile_size aanpassen
# tiler:
#   tile_size: 25.0
#   density_threshold: 100.0
```

### 3.2 Preprocessing pipeline uitvoeren (Python script)

```python
from pathlib import Path
from src.preprocessing.pipeline import PreprocessingPipeline
import numpy as np

# Laad je LiDAR data (bijvoorbeeld met laspy)
# points = ... (N x 8 array: X, Y, Z, range, intensity, R, G, B)

output_dir = Path("output/my_project")
output_dir.mkdir(parents=True, exist_ok=True)

pipeline = PreprocessingPipeline(
    output_dir=output_dir,
    provider="riegl_mms",
    tile_size=20.0,
    density_threshold=50.0
)

summary = pipeline.run(points, epsg=28992)
```

### 3.3 Preprocessing met Docker

```bash
docker run -v $(pwd)/data:/data -v $(pwd)/output:/output \
  lidarml-preprocess \
  python -c "from src.preprocessing.pipeline import PreprocessingPipeline; ..."
```

---

## 4. QA Visualisatie

### 4.1 BEV tiles visualiseren

```bash
python -m src.preprocessing.qa_bev_visualizer \
  --input output/demo_preprocessing/tiles_approved \
  --output output/demo_preprocessing/qa_visualizations
```

**Argumenten:**
- `--input`: Map met goedgekeurde tiles (`.parquet` bestanden)
- `--output`: Map voor visualisaties (PNG afbeeldingen)

### 4.2 Camera views visualiseren (demo mode)

```bash
python -m src.preprocessing.qa_cam_visualizer \
  --output output/demo_preprocessing/qa_camera \
  --demo
```

**Argumenten:**
- `--output`: Map voor camera visualisaties
- `--demo`: Gebruik synthetische demo data

### 4.3 Camera views visualiseren (echte data)

```bash
python -m src.preprocessing.qa_cam_visualizer \
  --input output/my_project/tiles_approved \
  --output output/my_project/qa_camera
```

---

## 5. ML Training - BEV Segmentatie

### 5.1 Dataset voorbereiden

Structuur:
```
data/dataset/bev/
├── images/
│   ├── train/
│   └── val/
└── masks/
    ├── train/
    └── val/
```

### 5.2 Configuratie aanpassen

Bewerk `configs/train_bev.yaml`:

```yaml
dataset:
  train_images: data/dataset/bev/images/train
  train_masks: data/dataset/bev/masks/train
  val_images: data/dataset/bev/images/val
  val_masks: data/dataset/bev/masks/val

model:
  n_classes: 5  # Pas aan naar aantal klassen

training:
  epochs: 50
  batch_size: 8
  learning_rate: 0.001
```

### 5.3 BEV model trainen

```bash
python -m src.ml.bev.train --config configs/train_bev.yaml
```

### 5.4 Training met Docker (GPU)

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/outputs:/workspace/outputs \
  lidarml-train \
  python -m src.ml.bev.train --config configs/train_bev.yaml
```

### 5.5 Training monitoren

Output wordt opgeslagen in `outputs/bev_unet.pt`

---

## 6. ML Training - Camera Detectie

### 6.1 Dataset voorbereiden (YOLO format)

Structuur:
```
data/dataset/cam/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### 6.2 data.yaml aanmaken

```yaml
path: data/dataset/cam
train: images/train
val: images/val

names:
  0: marking
  1: pole
  2: sign
  3: manhole
  4: guardrail
```

### 6.3 Configuratie aanpassen

Bewerk `configs/train_cam.yaml`:

```yaml
training:
  epochs: 100
  batch_size: 16
  imgsz: 640
```

### 6.4 Camera model trainen (Ultralytics YOLO)

```bash
python -m src.ml.cam.train --config configs/train_cam.yaml
```

### 6.5 Training met Docker

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/outputs:/workspace/outputs \
  lidarml-train \
  python -m src.ml.cam.train --config configs/train_cam.yaml
```

---

## 7. Inference & Voorspelling

### 7.1 BEV inference uitvoeren

```bash
python -m src.ml.bev.infer \
  --model outputs/bev_unet.pt \
  --input output/my_project/tiles_approved \
  --output output/predictions/bev
```

**Argumenten:**
- `--model`: Pad naar getraind model
- `--input`: Map met BEV tiles
- `--output`: Map voor voorspellingen

### 7.2 Camera inference uitvoeren

```bash
python -m src.ml.cam.infer \
  --model outputs/cam/weights/best.pt \
  --input output/my_project/camera_views \
  --output output/predictions/camera \
  --conf 0.25
```

**Argumenten:**
- `--model`: Pad naar YOLO model
- `--input`: Map met camera afbeeldingen
- `--output`: Map voor detecties
- `--conf`: Confidence threshold (0.0-1.0)

### 7.3 Batch inference met Docker

```bash
docker run --gpus all \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/outputs:/workspace/outputs \
  lidarml-train \
  python -m src.ml.bev.infer \
    --model outputs/bev_unet.pt \
    --input output/my_project/tiles_approved \
    --output output/predictions/bev
```

---

## 8. Object Extractie

### 8.1 Markering extractie

```python
from src.object_extraction.marking import extract_markings

markings = extract_markings(
    labeled_points="output/predictions/markings.laz",
    output_path="output/objects/markings.geojson"
)
```

### 8.2 Putdeksels extractie

```python
from src.object_extraction.manhole import extract_manholes

manholes = extract_manholes(
    labeled_points="output/predictions/manholes.laz",
    output_path="output/objects/manholes.geojson"
)
```

### 8.3 Masten extractie

```python
from src.object_extraction.mast import extract_masts

masts = extract_masts(
    labeled_points="output/predictions/masts.laz",
    output_path="output/objects/masts.geojson"
)
```

### 8.4 Borden extractie

```python
from src.object_extraction.traffic_sign import extract_traffic_signs

signs = extract_traffic_signs(
    labeled_points="output/predictions/signs.laz",
    output_path="output/objects/signs.geojson"
)
```

### 8.5 Vangrails extractie

```python
from src.object_extraction.guardrail import extract_guardrails

guardrails = extract_guardrails(
    labeled_points="output/predictions/guardrails.laz",
    output_path="output/objects/guardrails.geojson"
)
```

---

## 9. Wegmodel Generatie

### 9.1 Configuratie aanpassen

Bewerk `configs/roadmodel.yaml`:

```yaml
lane_width: 3.5
marking_width: 0.15
export_format: gpkg
```

### 9.2 Laangeometrie reconstrueren

```python
from src.roadmodel.lanes import reconstruct_lanes

lanes = reconstruct_lanes(
    markings="output/objects/markings.geojson",
    output_path="output/roadmodel/lanes.geojson"
)
```

### 9.3 CROW 96b classificatie toepassen

```python
from src.roadmodel.crow96b import classify_markings

classified = classify_markings(
    markings="output/objects/markings.geojson",
    output_path="output/roadmodel/markings_crow.geojson"
)
```

### 9.4 Dwarsprofielen genereren

```python
from src.roadmodel.profiles import generate_profiles

profiles = generate_profiles(
    points="output/my_project/scan_filtered.laz",
    centerline="output/roadmodel/lanes.geojson",
    output_path="output/roadmodel/profiles.geojson",
    interval=10.0  # meter
)
```

### 9.5 Complete wegmodel exporteren (GPKG)

```python
from src.roadmodel import export_roadmodel

export_roadmodel(
    lanes="output/roadmodel/lanes.geojson",
    markings="output/roadmodel/markings_crow.geojson",
    objects="output/objects/",
    output_path="output/roadmodel/wegmodel.gpkg",
    format="gpkg"
)
```

---

## 10. Tests Uitvoeren

### 10.1 Alle unit tests draaien

```bash
pytest tests/ -v
```

### 10.2 Specifieke module testen

```bash
pytest tests/preprocessing/test_pipeline_integration.py -v
```

### 10.3 Test met coverage rapport

```bash
pytest tests/ --cov=src --cov-report=html
```

Coverage rapport wordt opgeslagen in `htmlcov/index.html`

### 10.4 Alleen preprocessing tests

```bash
pytest tests/preprocessing/ -v
```

### 10.5 Tests in Docker uitvoeren

```bash
docker run -v $(pwd):/workspace lidarml-preprocess pytest tests/ -v
```

---

## 📝 Handige Tips

### Logging niveau aanpassen

```bash
export LOG_LEVEL=DEBUG
python examples/demo_preprocessing_pipeline.py
```

Mogelijke waarden: `DEBUG`, `INFO`, `WARNING`, `ERROR`

### Meerdere LAZ bestanden verwerken

```bash
for file in data/raw/laz/*.laz; do
  python -m src.preprocessing.pipeline --input "$file" --output "output/$(basename $file .laz)"
done
```

### GPU beschikbaarheid controleren

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Configuratie valideren

```bash
python -c "from src.utils.config import load_config; print(load_config('configs/preprocessing.yaml'))"
```

---

## 🐛 Troubleshooting

### Out of memory tijdens training

Verlaag `batch_size` in de training config:

```yaml
training:
  batch_size: 2  # Verlaag van 8 naar 2
```

### Docker GPU niet gevonden

Installeer NVIDIA Container Toolkit:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Tiles worden afgekeurd door QA

Controleer QA settings in `configs/preprocessing.yaml`:

```yaml
tiler:
  density_threshold: 50.0  # Verlaag threshold
```

### Module niet gevonden

Zorg dat je in de project root zit:

```bash
cd /pad/naar/lidarML_CL
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## 📚 Meer Informatie

- Zie `readme.md` voor architectuur en overzicht
- Zie `roadmap.md` voor toekomstige ontwikkelingen
- Zie `todo.md` voor actuele taken

---

**Versie:** 1.0
**Laatst bijgewerkt:** 2025-11-18
