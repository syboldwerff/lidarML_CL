# 🛣️ LiDAR Road Modeling Pipeline — Roadmap
*Volledig projectoverzicht, fases en deliverables*

---

## 🔍 Overzicht
Deze roadmap beschrijft alle logische vervolgstappen voor het bouwen van een complete Nederlandse wegobject-detectiepipeline:

- Preprocessing
- Dataset & annotatie
- ML-training
- 3D reconstructie
- Wegmodellering
- Validatie
- Opslag & integratie

De huidige preprocess-pipeline (BEV-6ch + axis-based camera-views + pixel→punt mapping) is voltooid. De volgende fases bouwen voort op dit fundament.

---

# 🧱 1. Preprocessing & QA (gecorrigeerde MMS, AHN6-niveau)
*Doel: gecorrigeerde, gekleurde scanauto-LiDAR valideren en tilen zodat BEV/cam direct gebruikt kan worden. Zware trajectstabilisatie blijft optioneel voor raw Riegl-exporten.*

### 1.1 Input- & metadata-checks
- CRS RD New + NAP bevestigd (of eenmalig converteren)
- RGB + intensity aanwezig en in verwacht bereik
- `meta_scan.json` met `crs_ok`, `has_rgb`, `has_intensity`, `bbox`, `provider`

### 1.2 Lichte cleaning
- Filter AOI-outliers (met buffer) en extreme Z-outliers
- Verwijder zwarte + zeer lage intensity-ruis
- Output: `scan_filtered.laz` + flag `basic_filter_ok`

### 1.3 Grond & hoogte-normalisatie
- Run PMF/CSF ground filter
- Labels: `is_ground`, `z_rel_ground`
- Flags per tile: `ground_ok` / `uncertain_ground`

### 1.4 Intensiteit & RGB-check
- Clamp intensity naar [1, 255]
- Check dat markering licht en asfalt donker blijft
- Optionele range-compensatie indien vendor dat niet doet; flag `intensity_ok`, `rgb_ok`

### 1.5 Tiling + QA per tile (20–25 m)
- Puntdichtheid, ground-fractie, intensity-histogram per tile
- Flags: `density_ok`, `intensity_ok`, `ground_ok`, `crs_ok`
- QA-outputs in `pre_tiles/*.parquet`

### 1.6 BEV/cam QA-visualisatie
- Gebruik enkel QA-goedgekeurde tiles in BEV/cam generators
- Visualiseer kanalen en mappings via kleine QA-scripts
- Multi-LAZ as-verwerking via `preprocess_cam_axis_multi.py` indien traject meerdere tiles doorkruist

### Deliverables
- `qa_bev_visualizer.py`
- `qa_cam_visualizer.py`
- Document: *“Preprocessing QA Findings”*

---

# 🗂️ 2. Class-schema & Dataset Planning (IMBOR/CROW)
*Doel: vastleggen wát we willen detecteren en in welk ML-formaat.*

### 2.1 Concept-classlist opstellen
Minimaal:
- Wegoppervlak
- Rijstrookmarkering (lang, dwars, as)
- Putdeksels
- Lichtmasten
- Borden (regulerend, bewegwijzering)
- Vangrail
- VRI-elementen
- Bermobjecten
- Trottoirband
- Talud / berm

### 2.2 Mapping naar IMBOR-codes
Voor elke klasse:
- IMBOR ID
- Beschrijving
- Geometrie-type (punt/lijn/object)
- Minimale detecteerbare resolutie

### 2.3 ML-formaat afspreken
- BEV → semantic segmentation
- Camera → object detection (YOLO/DETR)
- 3D → clustering per label

### Deliverables
- `classes.yaml`
- `IMBOR_mapping.md`
- `dataset_specification.md`

---

# 🖊️ 3. Annotatie & Mini Dataset
*Doel: minimaal werkende annotations → inference → 3D terugprojectie.*

### 3.1 Label-setup bepalen
- BEV-labels via polygon → raster (QGIS)
- Camera-labels via CVAT / LabelStudio / LabelMe

### 3.2 Eerste mini-dataset maken (AHN6)
Niet voor echte training — maar voor *end-to-end testen*. Labels zoals:
- Water
- Gras
- Verharding
- Gebouwen
- Bomen

### 3.3 Export pipelines voor labels
- BEV: `mask_{tile}.png`
- Camera: `annotations.json` (YOLO/COCO formaat)

### Deliverables
- `/labels/bev/`
- `/labels/camera/`
- `label_export_pipeline.md`

---

# 🎛️ 4. 3D Terugprojectie (Postprocessing v1)
*Doel: laten zien dat elk 2D label naar 3D punten kan worden omgezet.*

### 4.1 Module bouwen: BEV → 3D
Input:
- `tile_bev6.npz`
- `tile_mask.png`
- `*_bev_mapping.parquet`
Output:
- `tile_labelled_points.laz` of `.csv`

### 4.2 Module bouwen: Camera → 3D
Input:
- camera PNG
- YOLO/DETR bounding boxes
- mapping parquet
Output:
- 3D clusters van source LiDAR

### 4.3 3D clustering prototypes
- DBSCAN
- HDBSCAN
- Radius clustering

### Deliverables
- `project_bev_to_3d.py`
- `project_cam_to_3d.py`
- Document: *"3D Label Projection Tests”*

---

# 🤖 5. ML Training Container (GPU)
*Doel: een werkende minimale training/inference pipeline opzetten.*

### 5.1 Dockerfile.gpu
- CUDA 12
- PyTorch 2.x
- torchvision
- lightning
- numpy, opencv, pyarrow

### 5.2 BEV dataloader
- laadt `(6, H, W)` tiles
- koppelt aan maskers

### 5.3 Minimal BEV-model
- U-Net
- SegFormer (optioneel)

### 5.4 Train → Save → Predict tests
- Op basis van de AHN6 mini dataset

### Deliverables
- `docker/Dockerfile.train`
- `train_bev.py`
- `infer_bev.py`
- `models/checkpoints/`

---

# 🧬 6. Riegl Integratie
*Doel: preprocessing toepassen op echte mobiele LiDAR.*

### 6.1 Dichtheid-check & tiling
- puntdichtheid-profielen
- zichtlijnen
- datakwaliteit

### 6.2 BEV-6ch genereren
- hoge intensiteit-resolutie
- duidelijke markering & objectfeatures

### 6.3 Camera-views langs echte middenlijn
- betere visuele referentie
- ML-ready dataset

### 6.4 Eerste echte object-extractie
d.m.v. clustering + eerste ML modellen
- markering
- putdeksels
- masten
- borden

### Deliverables
- `riegl_integration_report.md`
- eerste geconverteerde dataset

---

# 🛠️ 7. Object Extractie (3D) v2
*Doel: echte 3D objecten bouwen op basis van ML labels.*

### 7.1 Markering → polyline
- connected components
- skeletonization
- rechte en gebogen segmenten splijten
- CROW 96b classificatie

### 7.2 Masten → cilindermodel
- RANSAC cylinder fit
- IMBOR-code koppelen

### 7.3 Putdeksels → ronding-detectie
- 3D cirkels / schijven
- hoogtecontrole

### 7.4 Borden → plane + reflectie
- bordpositie en oriëntatie afleiden

### Deliverables
- `/objects/`
- `object_reconstruction.md`

---

# 🗺️ 8. Wegmodel Generator v1
*Doel: alle objecten samenbrengen in een consistent NL-wegmodel.*

### 8.1 As-metrering koppelen aan objecten
- `s`-waarde per object
- dwarsoffset berekenen
- hoogteprofiel

### 8.2 Lane geometry reconstrueren
- rijstrookindeling
- middenlijn reconstructie
- rijstrookbreedtes

### 8.3 Wegmarkering volgens CROW 96b
- type
- subtype
- reflectieklasse
- lengte / patroon

### 8.4 Exportformaten
- GeoJSON
- GPKG (IMBOR-vriendelijk)
- Lanelet2
- OpenDRIVE (optioneel)

### Deliverables
- `wegmodel_exporter.py`
- `wegmodel_schema_v1.gpkg`

---

# 📊 9. Validatie & Kwaliteitscontrole
*Doel: betrouwbare QA bovenop de automatische pipeline.*

### 9.1 CROW-validatie
- markering breedte
- markering afstand
- objectpositie tot middenlijn

### 9.2 Densiteit & dekking
- heatmaps
- hole detection

### 9.3 Model confidence checks
- outliers
- inconsistenties

### Deliverables
- `qa_reports/`
- `crow_validation.md`

---

# 🏁 10. Einddoel — Volledig Geautomatiseerde Wegmodellering
Een complete keten van ruwe mobiele Riegl-LAZ naar een volledig wegmodel:

1. Preprocess (BEV + camera)
2. Annotatie / training
3. ML-detectie
4. 3D-terugprojectie
5. Objectreconstructie
6. Wegmodelgeneratie
7. IMBOR-export
8. Validatie
9. Publiceerbare dataset & rapport

---

Deze roadmap vormt de leidraad voor de komende ontwikkelfases en geeft een gestructureerd pad naar een professionele, robuuste en reproduceerbare LiDAR-wegmodellering pipeline.

