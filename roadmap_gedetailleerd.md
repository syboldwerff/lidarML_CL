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

# 🧱 1. Preprocessing & QA (Gecorrigeerde LiDAR, AHN6-niveau)
*Doel: van een gecorrigeerde, gekleurde MMS-puntenwolk naar QA-gelabelde tiles die direct de BEV- en camera-preprocessor in kunnen.*

We gaan uit van een leverancier die al GNSS/IMU-correctie, kleur en een bruikbaar CRS (bijv. RD New + NAP) levert — vergelijkbaar met AHN6-kwaliteit. De zware trajectstabilisatie en intensieve ghost-cleaning blijven optioneel voor het moment dat je ook ruwe Riegl-export wilt ondersteunen.

Wat wordt optioneel t.o.v. de ruwe pipeline:
- Trajectherstel & stabilisatie (alleen nodig voor echte raw Riegl/Applanix-export).
- Zware ruis-/ghost-filtering (kan veel lichter, leverancier heeft al opgeschoond).
- Volledige CRS-normalisatie (alleen een check zolang input al RD/NAP is).
- Puntdichtheidsuniformisering per scanlijn (alleen meten en flaggen is genoeg voor BEV).

---

## 1.1 Inlees- & metadata-controle
**Check:**
- CRS is RD New + NAP (of eenmalig converteren).
- RGB-kanalen aanwezig (0–255).
- Intensity-range in verwacht bereik.

**Output:** `meta_scan.json` met `crs_ok`, `has_rgb`, `has_intensity`, `point_count`, `bbox`, `provider` om te borgen dat de dataset “AHN6-achtig” oogt.

---

## 1.2 Basisfiltering (lichte cleaning)
Geen agressieve smoothing; alleen evidente troep verwijderen.

**Taken:**
- Punten ver buiten de area of interest (met buffer) verwijderen.
- Extreme Z-outliers wegfilteren (bijv. > +100 m of < –20 m NAP).
- Punten met volledig zwarte RGB én extreem lage intensiteit verwijderen.

**Output:** `scan_filtered.laz` + flag `basic_filter_ok`.

---

## 1.3 Grondsegmentatie & hoogte-normalisatie (licht)
Geen volledige DTM; alleen labels en relatieve hoogte voor wegprofielen en taluds.

**Taken:**
- Ground filter draaien (PMF/CSF of PDAL ground).
- Per punt `is_ground` markeren.
- Kolom `z_rel_ground = z - z_ground_local` toevoegen.

**Flags:** `ground_ok` / `uncertain_ground` (per tile).

---

## 1.4 Intensiteit & RGB-normalisatie (check + lichte correctie)
Voor consistente markeringen zonder agressieve correcties.

**Taken:**
- Histogram van intensity clampen naar [1, 255].
- Controleren of markering de lichte wolk vormt en asfalt donkerder blijft.
- Optioneel: simpele range-compensatie wanneer de leverancier dat niet doet.

**Flags:** `intensity_ok`, `rgb_ok`.

---

## 1.5 Tiling + QA per tile (20×20 m)
Brug naar de bestaande BEV-pipeline.

**Taken:**
- `scan_filtered.laz` snijden in 20×20 m (of 25×25 m) tiles in RD-grid.
- Per tile meten:
  - puntdichtheid (pts/m²)
  - fractie ground vs. non-ground
  - intensiteit-histogram
  - aanwezigheid van genoeg punten in wegvlak (voor markering)

**Output:** `pre_tiles/*.parquet` met flags `density_ok` / `density_hole`, `intensity_ok`, `ground_ok`, `crs_ok` (globaal).

---

## 1.6 QA-flagging & selectie voor BEV/cam
BEV- en cam-generators gebruiken alleen tiles met `density_ok`, `intensity_ok`, `ground_ok`, `crs_ok`. Tiles met issues gaan naar `qa_review/` of worden als hard negatives opgeslagen.

**Resultaat Stap 1:**
- `scan_filtered.laz` (of de originele `corrected_points.laz` als filtering minimaal blijft).
- `meta_scan.json` met inputchecks.
- `pre_tiles/*.parquet` met QA per tile en flags voor selectie.

---

# 🗂️ 2. Class-schema, Dataset Planning & Tile-Selectie (QA-aware)
*Doel: bepalen welke objecten we willen herkennen, en hoe we daarvoor een schone, consistente, QA-gestuurde dataset opbouwen.*

Deze fase werd oorspronkelijk ontworpen voor reeds opgeschoonde data.  
Met de nieuwe ruwe-data preprocessing voegen we nu een kritische stap toe:  
**alle training-, validatie- en annotatiedata moet QA-geslaagd zijn.**  
Slechte tiles worden automatisch uitgesloten of apart gelabeld.

---

## 2.1 Class-schema volgens Nederlandse standaarden (IMBOR/CROW)
Voor elk detecteerbaar wegobject definiëren we nu:

- **IMBOR-klasse**
- **Beschrijving**
- **Geometrie-type** (punt, lijn, vlak)
- **Detectieschaal / minimale resolutie**
- **Kanaalafhankelijkheid** (BEV, camera of beide)
- **3D reconstructievereisten** (cilinder, polyline, vlak)

Voorbeeld basisklassen:
- Rijstrookmarkering (as- en kantmarkering)
- Dwarsmarkeringen (stopstrepen, haaientanden)
- Putdeksels
- Lichtmasten
- Borden (regulerend / informatie)
- Vangrail
- Trottoirband
- Bermobjecten
- Talud / bermoppervlak

**Output:** `classes.yaml` + `IMBOR_mapping.md`

---

## 2.2 Datasetstructuur op basis van QA-tiles
In plaats van willekeurige tiles gebruiken we nu:

- Alleen tiles met:
  - `density_ok`
  - `intensity_ok`
  - `ground_ok`
  - `crs_ok`
  - optioneel: `trajectory_ok` / `roll_pitch_ok` wanneer je ook raw-trajecten verwerkt

- Tiles met QA-waarschuwingen komen in:
  - **val_set_qa** (om robuustheid te testen)
  - **hard-negative pool** (voor later)

**Voordeel:**  
Het ML-model leert op **technisch consistente** voorbeelden en blijft schoon van bias door slechte inputdata.

---

## 2.3 Selectie van BEV-tiles voor annotatie
Per QA-geslaagde tile hebben we al:

- BEV 6-kanaals NPZ  
- BEV PNG  
- Pixel→punt mapping

Annotatie gebeurt uitsluitend op deze QA-positieve tiles.

Extra checks:
- minimaal aantal punten per pixel  
- intensiteit histogram in verwacht bereik  
- geen tilt/stabilisatie-flags  

**Output folders:**
/dataset/bev/images/
/dataset/bev/masks/
/dataset/bev/mapping/

---

## 2.4 Selectie van camera views voor annotatie
Camera views worden nu ook QA-aware:

- Alleen gegenereerd langs trajectsegmenten met geldige QA (géén drift, géén instabiele roll/pitch)
- Views met slechte ondergrond of density_hole worden automatisch uitgesloten

We genereren:
- 6 richtingstegels per positie
- PNG’s + mapping per view

**Output folders:**
/dataset/camera/images/
/dataset/camera/annotations/
/dataset/camera/mapping/

---

## 2.5 Annotatiestrategie (BEV + camera)
Een gecombineerde annotatiestrategie:

### BEV (2D semantic segmentation)
- Ideaal voor:
  - markering
  - putdeksels
  - trottoirband
  - wegoppervlak
  - berm / talud

Annotatietools:
- QGIS (polygon → rasterized mask)
- LabelStudio (custom BEV plugin)
- CVAT BEV mode

### Camera (object detection)
- Ideaal voor:
  - lichtmasten
  - borden
  - verkeerslichten
  - kleine bermobjecten

Annotatietools:
- CVAT
- LabelStudio
- YOLO-web tools (optioneel)

---

## 2.6 Minimale trainingsdataset bepalen (kwaliteitsgestuurd)
We bouwen eerst een **kleine maar extreem consistente** dataset:

- ± 200–400 BEV-tiles (clean)
- ± 300–500 camera views (clean)
- 5–10 km variatie in omgevingstypen:
  - stedelijk
  - buitenweg
  - snelweg
  - industrie

De dataset wordt bewust klein gehouden om:

- snel iteratieve modellen te testen  
- annotatie-inspanning laag te houden  
- fouten snel te lokaliseren  

Later breiden we uit met QA-hardcases.

---

## 2.7 Label-export pipelines
Alle QA-positieve annotaties worden automatisch geëxporteerd naar ML-ready format:

**BEV export:**
- `mask_{tile_id}.png`
- `tile_{id}.npz`  
- mapping parquet

**Camera export:**
- YOLOv8: `.txt` files per view  
- COCO JSON (optioneel)
- mapping parquet

**Output:** `dataset_specification.md`

---

## 2.8 Relatie tussen QA-data en annotatieportaal
De webportal kan:

- QA-negatieve tiles tonen als **“Bad / Review”**  
- QA-positieve tiles tonen als **“Good / Ready for Annotation”**  
- Handmatig slechte tiles alsnog herprocessen  
- Annotaties direct koppelen aan tile-ID en flags

Dit maakt een menselijke QA-loop mogelijk zonder dataschade.

---

## 2.9 Resultaat van Stap 2
Aan het einde van deze fase heb je:

- een formele NL-classlist (IMBOR/CROW)
- een complete, QA-gestuurde datasetspecificatie
- een selectie van alleen technisch solide tiles
- een kleine maar hoogwaardige trainingsset
- mapping-bestanden voor 3D terugprojectie
- correcte annotaties in ML-formaten

Het ML-leesbare fundament is nu volledig gelegd.

---

# 🖊️ 3. Annotatie & Mini Dataset (QA-gestuurd)
*Doel: een kleine maar hoogwaardige dataset maken om end-to-end testen mogelijk te maken — inclusief 2D labels, 3D terugprojectie en eerste inferenties.*

Deze fase draait niet om grote volumes data, maar om **kwaliteit en correctheid**.  
Alle annotatie gebeurt uitsluitend op tiles/views die door **Stap 1 (Preprocessing & QA)** als *goed* zijn beoordeeld.

---

## 3.1 Doel: end-to-end testset bouwen
Voordat we een grote dataset maken, creëren we een *Mini E2E Dataset* waarmee we de volledige pipeline van:

1. BEV/camera generatie  
2. annotatie  
3. training  
4. inferentie  
5. 3D terugprojectie  

kunnen valideren.

De mini-dataset moet:
- klein zijn (gemaakt in 1–2 avonden annoteren)
- heel betrouwbaar zijn
- meerdere scenario’s bevatten (stad, buitenweg, snelweg)

---

## 3.2 Selectie van QA-positieve tiles
We gebruiken alleen tiles waar:

- `density_ok`
- `intensity_ok`
- `ground_ok`
- `crs_ok`
- optioneel: `trajectory_ok` / `roll_pitch_ok` wanneer raw-trajecten in scope zijn

Tiles met één of meer waarschuwingen worden óf genegeerd, óf toegevoegd aan een aparte:
- `dataset/hard_negatives/`
- `dataset/qa_review/`

Dit voorkomt dat het ML-model verkeerde patronen leert.

---

## 3.3 BEV Annotatie (2D semantic segmentation)
BEV is perfect voor objecten die structureel op/naast het wegoppervlak liggen.

Annoteren als *maskers* voor:
- rijstrookmarkeringen (as / kant / verdrijvingsvlak)
- dwarsmarkering (stopstrepen, haaientanden)
- putdeksels
- trottoirband
- asfalt / berm / talud
- verkeersgeleiders (bijvoorbeeld vluchtheuvels)

### Workflow:
- BEV-PNG openen in QGIS / LabelStudio / CVAT
- polygonen tekenen
- rasteriseren tot `mask_tileID.png`
- mask heeft exact dezelfde resolutie als de BEV PNG/NPZ

### Output:
- `/dataset/bev/images/`
- `/dataset/bev/masks/`
- `/dataset/bev/mapping/`

Mapping is later nodig voor 3D-terugprojectie.

---

## 3.4 Camera Annotatie (object detection)
2D camerabeelden zijn ideaal voor:
- lichtmasten  
- verkeersborden  
- verkeerslichten  
- kleine bermobjecten  
- bebording die in BEV slecht zichtbaar is

Annoteren in:  
- CVAT  
- LabelStudio  
- YOLO annotators  

Elke view heeft:
- PNG  
- YOLO txt of COCO JSON  
- mapping parquet voor 3D terugprojectie  

**Alleen camera’s gegenereerd van QA-positieve segments worden gebruikt.**

### Output:
- `/dataset/camera/images/`
- `/dataset/camera/labels/`
- `/dataset/camera/mapping/`

---

## 3.5 3D Terugprojectie — validatie van annotaties
De mini dataset wordt direct getest met:

- `project_bev_to_3d.py`  
- `project_cam_to_3d.py`

Doelen:
- controleren of polylijnen van markeringen logisch op het wegvlak liggen  
- checken of putdeksels in de grondzak vallen  
- checken of masten consistente cilinders vormen  
- detecteren van annotatiefouten (“zwevende objecten”, offset door perspectief)  

Tiles die 3D-annotatiefouten veroorzaken → flag: `needs_review`.

Resultaat is een **volledig sluitende 2D ↔ 3D dataset**, essentieel voor serieuze training.

---

## 3.6 Minimale datasetomvang
We maken de dataset **bewust klein maar divers**:

### BEV:
- 150–250 tiles  
- mix van stedelijk / buitenweg / snelweg  
- gefocust op markering + putdeksels + bermovergangen  

### Camera:
- 300–500 views  
- gefocust op borden + masten  
- gegarandeerd QA-positief

### Hard-negatives:
- 50–100 tiles die mislukkingen bevatten (bewust niet gelabeld)  
- gebruikt in validatiefase later

---

## 3.7 Dataset-export pipelines
Voor ML-training zijn consistente formaten vereist:

### BEV-export:
- `tileID.npz` (6 channels tensor)
- `mask_tileID.png`
- mapping parquet

### Camera-export:
- `viewID.png`
- `viewID.txt` (YOLO) of COCO JSON  
- mapping parquet

Daarnaast:
- train/val/test splits  
- versiebeheer (`dataset_v1/`, `dataset_v2/`)  
- checksums voor dataset-integriteit  

**Output:** `label_export_pipeline.md`

---

## 3.8 Portal-integratie (Good/Bad labeling)
De annotatieportal krijgt een standaard pipeline:

- BEV-tile of camera view tonen  
- QA-flags automatisch zichtbaar  
- “Good” → naar annotatie of training  
- “Bad” → markeren voor herprocessing  
- metadata wordt opgeslagen in:  
  `/portal_meta/tileID.json`

Bij **Bad** tiles kan de preprocessing automatisch opnieuw draaien.

---

## 3.9 Resultaat van Stap 3
Na deze stap is er:

- een technisch perfecte mini-dataset  
- QA-positieve BEV- en camera labels  
- mapping→3D getest  
- hard-negative bibliotheek  
- export pipelines in ML-formaat  
- basis voor eerste modeltrainingen  

De volledige pipeline kan hiermee voor het eerst **end-to-end getest** worden.

---

# 🎛️ 4. 3D Terugprojectie (Postprocessing v1, QA-driven)
*Doel: elke 2D-annotatie of ML-prediction betrouwbaar terugprojecteren naar de originele 3D punten, zodat objecten later gereconstrueerd kunnen worden.*

Dankzij de mapping-bestanden uit de preprocessing (BEV + camera) kunnen alle 2D labels exact aan de juiste LiDAR-punten gekoppeld worden.  
Dit maakt de hele keten **geometrisch sluitend** — een harde eis voor IMBOR/CROW-conforme objectextractie.

---

## 4.1 BEV → 3D projectiemodule
BEV-tiles komen met:
- `(6, H, W)` tensor (`tile.npz`)
- `tile.png`
- `tile_mapping.parquet` (bevat voor elke pixel: point_ids, afstand, hoogte, lokale tilepositie)

De BEV-3D projector:

1. leest het mask (`mask_tileID.png`)
2. haalt per relevante pixel de bijbehorende point_ids op
3. groepeert alle punten per labelklasse
4. schrijft een gestructureerd bestand weg:

**Output per tile:**
tileID_labelled_points.laz
tileID_labelled_points.parquet
tileID_polylines_preview.gpkg (optioneel)

**Doelen:**
- polylijnen voor markeringen reconstrueren
- clusters maken voor putdeksels
- controle of labels overeenkomen met fysiek verwachte hoogte

**QA:**
- misprojecties → flag `projection_inconsistent`
- zwevende punten → flag `height_mismatch`

---

## 4.2 Camera → 3D projectiemodule
Camera views hebben:
- PNG beeld
- YOLO/COCO annotaties of ML-predictions
- `view_mapping.parquet`

De camera-3D projector:

1. leest bounding boxes / masks per view
2. haalt alle LiDAR-punten op die binnen de box vallen  
   (mapping bevat pixel→puntindex)
3. groepeert punten per object
4. legt tijdelijk metadata vast:
   - kijkrichting
   - afstand
   - segment (chainage)
   - viewID → useful for multi-view fusion

**Output per view:**
viewID_clusters.parquet
viewID_clusters_preview.laz

**Doelen:**
- masten reconstrueren als cilindrische clusters
- borden reconstrueren als vlakjes
- reflecterende objecten identificeren (hoge intensiteit)

**QA:**
- te weinig punten in box → `insufficient_points`
- grote dieptevariatie → `depth_inconsistent`
- afwijkende hoogte → `height_outlier`

---

## 4.3 Fusion van meerdere views + BEV
Veel objecten worden vanuit verschillende camera-views én in BEV gezien.  
We bouwen daarom een **fusion layer**:

- punten van BEV en camera worden samengevoegd
- duplicates (identieke point_ids) worden gefilterd
- labelconflicten tussen BEV en camera worden gelogd

**Fusion output:**
objectID_raw_points.laz
objectID_metadata.json
fusion_report_tileID.md

---

## 4.4 Clustering als voorbereidende stap voor objectextractie
Voor elk gelabeld object worden 3D clusters gemaakt:

**Voorwaarden:**
- 3D DBSCAN
- grenswaarden per objecttype (radius, min_samples)
- opsplitsing in meerdere objecten indien nodig

Toepassingen:
- masten → één cluster per paal
- borden → vlak-detectie (plane fit)
- putdeksels → cirkelvormige clusters
- markering → connected components → polylijntracking

Clusters vormen de input voor **Sectie 7: Object Extractie v2**.

---

## 4.5 3D debugging en visualisatie previews
Per tile en view worden preview-bestanden gemaakt om de pipeline visueel te controleren:

- `*_clusters_preview.laz`
- `*_labelled_points.laz`
- `tileID_polylines_preview.gpkg`

Deze bestanden worden gebruikt in:
- QGIS voor polylinecontrole  
- CloudCompare voor puntclusters  
- Portal voor visuele kwaliteitscontrole  

---

## 4.6 QA-integratie in 3D-projectie
3D-terugprojectie introduceert een nieuwe set QA-flags:

- `projection_inconsistent` (BEV polygon → verspreide punten)
- `depth_inconsistent` (camera box → grote afstandsrange)
- `height_mismatch` (object zweeft of zinkt)
- `duplication_conflict` (camera/BEV label mismatch)
- `multiobject_overlap` (clustering split noodzakelijk)
- `insufficient_points`

Alle flags worden per tile en per object opgeslagen in:
/qa/project3d/tileID.json
/qa/project3d/viewID.json

Tiles/views met ernstige fouten gaan naar:
- `qa_review/`
- of worden automatisch gemarkeerd als *Bad* in de portal.

---

## 4.7 Resultaat van Stap 4
Aan het einde van deze fase heb je:

- een robuuste BEV → 3D projectie
- een robuuste camera → 3D projectie
- gefuseerde 3D objectpunten (multiview + BEV)
- clusters voor alle objecttypes
- preview-lagen voor QA
- een volledige, geometrisch sluitende dataset

Dit vormt de input voor:
- Sectie 5 (ML training container)
- Sectie 7 (3D Object Extractie)
- Sectie 8 (Wegmodel Generator)

---

# 🤖 5. ML Training Container (GPU, QA-aware, Reproducible)
*Doel: een volledig Docker-gebaseerde GPU-trainingsomgeving bouwen voor BEV-segmentatie en camera-detectie, inclusief dataloaders, logging, QA-integratie en retraining-ondersteuning vanuit de portal.*

Deze omgeving maakt het mogelijk om:
- BEV-segmentatiemodellen te trainen
- camera-objectdetectie te trainen (YOLO/DETR)
- inference te draaien op nieuwe tiles
- modellen automatisch opnieuw te trainen op basis van Good/Bad feedback uit de annotatieportal  
- reproduceerbare experimenten bij te houden in versies

---

## 5.1 CUDA-gebaseerde training Dockerfile
We bouwen een dedicated GPU-container:

**Basis:**
- `nvidia/cuda:12.x-cudnn8-runtime-ubuntu22.04`
- Python 3.11
- PyTorch 2.x + CUDA
- torchmetrics
- lightning
- opencv-python-headless
- pyarrow, pandas
- tqdm, rich
- albumentations / imgaug (camera augmentations)
- rasterio (optioneel)
- laspy (voor spot-checks)
- pydantic (config validatie)

**Bestanden:**
docker/Dockerfile.train
docker-compose.train.yml

---

## 5.2 Dataloaders: BEV en camera (QA-aware)
### BEV Dataloader
Laadt:
- `tileID.npz`  (6 kanalen)
- `mask_tileID.png`
- `tile_mapping.parquet` (optioneel ter debug)
- QA-metadate (`tileID.json`)

**Eisen:**
- reject tiles waar QA ≠ volledig OK  
- logging van skipped tiles per epoch  
- automatische oversampling per klasse (class balancing)

---

### Camera Dataloader
Laadt:
- `viewID.png`
- YOLO `.txt` of COCO `.json`
- mapping parquet
- QA-metadate

**Functies:**
- geavanceerde augmentaties (weather, blur, exposure)
- skip views met QA-warnings
- class-balancing & per-object type sampling

---

## 5.3 Modellen
We beginnen met minimal-viable modellen:

### BEV-segmentatie
- U-Net (baseline)
- SegFormer-B0 (productierichting)

Input: `(6, H, W)`  
Output: segmentation mask

### Camera-detectie
- YOLOv8n / v8s (lightweight)
- of DETR-tiny voor robuustheid bij variabele lichtcondities

Later uitbreidbaar.

---

## 5.4 Training pipelines
Elke trainingrun is volledig configureerbaar via:

configs/train_bev.yaml
configs/train_cam.yaml

Met:
- batch size
- augmentations
- optimizer
- lr-schedule
- checkpointpad
- datasetversie

**Opslagstructuur:**
models/
bev/
run_YYYYMMDD_hhmm/
checkpoints/
logs/
config.yaml
camera/
run_YYYYMMDD_hhmm/
...

---

## 5.5 Logging, metrics & QA-integratie
Training en evaluatie logt:

- per-klasse IoU (BEV)
- precision/recall (camera)
- confusion matrix
- tile-level QA-statistieken:
  - skipped by QA  
  - borderline QA tiles  
  - mislukkingen bij inference  
- best/worst samples (automatisch opgeslagen PNG’s)

**Tools:**
- TensorBoard
- wandb (optioneel)
- eigen HTML-logs per run

---

## 5.6 Inference module (voor portal & batch)
We maken een generieke inferencer:
infer_bev.py
infer_cam.py

Functies:
- inference op losse tiles
- inference op volledige trajecten
- opslaan van:
  - prediction PNG’s
  - confidences
  - predicted mask → 3D mappingentry

**Flags:**
- inference mag QA-warnings negeren  
- portal gebruikt inference-resultaten om Good/Bad aan te geven  

---

## 5.7 Fine-tuning & Retraining flow (via portal)
De portal kan aangeven:

- tile/view = Good  
- tile/view = Bad  
- tile/view = Retain (herannoteren)
- tile/view = Remove

We bouwen een retraining-API:
python train_bev.py --retrain-from portal_feedback.json

Flow:
1. portal → laat gebruiker Good/Bad markeren  
2. `portal_feedback.json` bevat tileID’s + acties  
3. trainingcontainer laadt alleen Good tiles  
4. Bad tiles komen in een hard-negative buffer  
5. nieuwe checkpoints worden opgeslagen

Dit maakt een **mens-in-de-loop** ML-cyclus mogelijk.

---

## 5.8 Versiebeheer (datasets + modellen)
Elk experiment wordt vastgelegd in een versie:

dataset_v1/
dataset_v2/
models_v1/
models_v2/


Met:
- changelogs
- checksums
- config freeze
- tileslijst

Hiermee is alle research reproduceerbaar en exporteerbaar.

---

## 5.9 Resultaat van Stap 5
Aan het einde van deze stap:

- GPU-trainingscontainer is volledig operationeel
- BEV en camera dataloaders zijn QA-aware
- baseline modellen zijn trainbaar en inference-ready
- Good/Bad feedback uit portal kan training direct beïnvloeden
- inference + retraining vormen een gesloten ML-loop
- alle runs zijn reproduceerbaar en gelogd

---

# 🧬 6. Riegl Integratie (Mobiele LiDAR, QA-driven)
*Doel: de volledige preprocessing & ML-pipeline toepassen op echte mobiele Riegl-data uit de scanauto, inclusief trajectcorrectie, densiteitsanalyse, BEV/camera generatie en kwaliteitsrapporten.*

Tot nu toe werkten alle stappen met AHN6 en testdata.  
In deze fase sluit de hele pipeline aan op real-world mobiele LiDAR afkomstig uit de scanauto (Riegl + Applanix/NovAtel).

Deze stap vormt de brug tussen experiment → productie.

---

## 6.1 Import van mobiele Riegl data (vendor-corrected → preproc)
We starten vanaf een **leverancier-gecorrigeerde MMS-export** (GNSS/IMU gestabiliseerd, RD/NAP, RGB/intensity aanwezig). De optionele ruwe trajectketen blijft beschikbaar voor het moment dat je echte Riegl-swaths wilt ondersteunen, maar is niet de default.

De **Stap 1 preprocessing (lichte variant)** draait hier voor het eerst end-to-end op echte mobiele data.

**Output:**
- `scan_filtered.laz` (of direct `corrected_points.laz` bij minimale filtering)
- `meta_scan.json`
- `pre_tiles/*.parquet`
- QA flags per segment & tile (incl. optioneel `trajectory_corrected.json` wanneer de raw-route wordt gebruikt)

---

## 6.2 Puntdichtheidprofielen & dekking
Mobiele LiDAR is grillig: puntdichtheid varieert met bochten, snelheid, asfaltreflectie, en verkeerssituaties.

Onze densiteitsmodule levert:
- heatmaps van dekking
- plots van density per meter traject
- detectie van:
  - density holes  
  - glare/reflection zones  
  - obstructies door verkeer

**Flag:** `density_hole` → deze tiles worden automatisch gepasseerd of herprocessen.

---

## 6.3 BEV-generatie op echte Riegl data
Alle QA-positieve tiles worden omgezet naar BEV-6ch:

- z_max  
- intensiteit_mean  
- r_mean / g_mean / b_mean  
- density  

Inclusief:
- gestabiliseerde roll/pitch (uit trajectcorrectie)
- uniformisatie van intensiteit (range compensation + flattening)

**Output per tile:**
tileID.png
tileID.npz
tileID_mapping.parquet
tileID.json (QA + metainfo)

---

## 6.4 Cameraviews genereren langs de middenas
Gebruikmakend van een asbestand (CSV met X/Y[/Z]) of automatisch afgeleide baanlijn.

Per view:
- 6 richtingen (front, fl, fr, back, bl, br)
- 1920×1080 rendering
- alleen punten binnen search-radius
- QA aangevuld met:
  - roll/pitch stabiliteit
  - zichtbaarheid
  - diepte-range controle

**Output per view:**
viewID.png
viewID_mapping.parquet
viewID.json (QA + pose + metadata)

---

## 6.5 Eerste inferentie op echte Riegl data
Met de modellen uit Sectie 5:

- BEV-segmentatie  
- Camera-objectdetectie  

We draaien voor het eerst inference op echte mobiele data om:
- markeringen te detecteren  
- putdeksels te vinden  
- masten en borden in 3D te clusteren  

**Output:**
pred_bev_tileID.png
pred_bev_mask_tileID.png
pred_camera_viewID.png


Alle predictions worden automatisch naar 3D teruggeprojecteerd via Sectie 4.

---

## 6.6 3D fusie & kwaliteitscontrole (real-world)
De volgende checks zijn essentieel:

### 3D checks:
- markeringen liggen op het wegvlak  
- masten vormen cilindrische clusters  
- borden vormen vlakjes  
- putdeksels liggen op/in de grond  

### Consistentiechecks:
- BEV & camera prediction moeten overeenkomen  
- multi-view fusion moet één object opleveren  
- inconsistenties worden gelogd

**Output:**
riegl_integration_report.md
clusters_preview.laz
fusion_report_.md
qa/riegl/.json

---

## 6.7 Detectie van kritische dataproblemen
Deze fase vindt structurele fouten in Riegl-opnames:

- slechte GPS-ontvangst (bruggen, tunnels)
- instabiel traject bij lage snelheid
- reflecterende verkeerssituaties (water, glas)
- regenbollen / vocht op lens
- vibratie in het voertuig

De pipeline genereert rapporten die vertellen of heropname, extra smoothing of handmatige correctie nodig is.

---

## 6.8 Resultaat van Stap 6
Aan het einde van deze fase is het systeem voor het eerst **getest op echte mobiele Riegl LiDAR**.

Je hebt:

- een volledig opgeschoonde, gecorrigeerde, gestabiliseerde puntenwolk  
- QA per tile/view  
- BEV-6ch + camera views + mapping  
- inference-resultaten op echte data  
- 3D clusters en fusie van detecties  
- kwaliteitsrapporten voor engineeringbeslissingen  
- een keten die klaar is voor **Object Extractie (Sectie 7)** en **Wegmodellering (Sectie 8)**  

Dit is het kantelpunt waar het systeem van experiment → operatie gaat.

---

# 🛠️ 7. Object Extractie (3D) v2 – Geometry-first, IMBOR-ready
*Doel: uit de gefuseerde 3D puntenwolken echte objecten extraheren: markeringen, masten, borden, putdeksels, trottoirband, vangrail, bermobjecten — alles conform IMBOR/CROW-structuur.*

Waar Sectie 4 de 3D terugprojectie regelt, bouwt deze fase de **echte objecten**.  
Alle objectextractie is gebaseerd op de QA-gefilterde, 3D-gefuseerde dataset uit Sectie 6.

Dit is de stap waarin de pipeline tastbare engineeringfeatures oplevert.

---

## 7.1 Voorverwerking: objectclusters klaarzetten
Uit Sectie 4 hebben we per tile/view:

- BEV→3D clusters
- camera→3D clusters
- gefuseerde clusters

We starten hier met:

- het samenvoegen van puntgroepen
- filtering van noise clusters
- normalisatie van attributen (intensiteit, Z, pointcount, hoogtevariatie)

**Output:**  
`object_candidates.parquet` — een lijst met ruwe objectclusters.

---

## 7.2 Extractie van wegmarkeringen (lijnen + patronen)
Markering is de meest geometriegevoelige categorie.

### Workflow:
1. **Connected components** op BEV-maskers  
2. 3D-punten bundelen → segmenten  
3. **Skeletonization** om de middellijn te berekenen  
4. Segmenten splitsen:
   - rechte delen
   - boogdelen
   - patronen (blokmarkering, verdrijvingsvlak, haaientanden)

### Attributen:
- breedte
- oriëntatie
- 3D curve
- lengte
- type (CROW 96b)

### Output:
markering/
tileID_markering.gpkg
tileID_markering.json

Inclusief:
- polyline geometrie
- CROW-markeringstype
- confidence score
- QA flags

---

## 7.3 Extractie van putdeksels (cirkels / schijven)
Putdeksels zijn kleine maar betrouwbare 3D objecten.

### Methode:
- 3D clustering (DBSCAN)
- hoogte-analyse: plateau in Z
- horizontale circular fit (RANSAC circle)
- check: diameter binnen IMBOR-range (meestal 400–600 mm)
- check: ligt op wegvlak (± 3 cm)

### Output:
putdeksels/
tileID_putdeksels.gpkg
tileID_putdeksels.json

---

## 7.4 Extractie van lichtmasten (cilinders)
Masten zijn perfect voor cilindrische fitting.

### Methode:
1. cluster extractie
2. RANSAC cylinder fit  
3. bepalen:
   - voetpunt (X,Y,Z)
   - hoogte (Z_range)
   - schachtdiameter
   - oriëntatie (verticaliteit)

### QA:
- cilinder residu te hoog → `bad_cylinder_fit`
- te weinig punten → `insufficient_points`

### Output:
masten/
mastID.gpkg
mastID.json

---

## 7.5 Extractie van verkeersborden (vlakken)
Borden zijn kleine vlakjes met hoge reflectie.

### Methode:
- BEV + camera fusion
- plane fitting via RANSAC
- label normaliseren:
  - oriëntatie → richting verkeer
  - hoogteklasse (CROW conform)
- clustering van meerdere zichtbare borden (meerdere faces)

### Output:
borden/
bordID.gpkg
bordID.json

---

## 7.6 Extractie van vangrail (lineaire objecten)
Vangrail is een lineair 3D object dat niet altijd goed segmentatief te zien is.

### Methode:
- BEV-detectie + hoogtefilter  
- cluster in lange stroken  
- 3D polyline fit  
- regularisatie:
  - gladheid
  - minimal turn radius  
- optie: ondersteunende palen detecteren (verticale mini-cilinders)

### Output:
vangrail/
segmentID.gpkg
segmentID.json

---

## 7.7 Extractie van trottoirband (randen)
Trottoirband is een hard lineair object.

### Methode:
- BEV-signatuur (intensiteit + hoogte)  
- 3D cross-section detectie  
- polyline reconstructie  
- detectie van insteekgebieden / opritten  

### Output:
band/
bandID.gpkg
bandID.json

---

## 7.8 Bermobjecten (diverse kleine objecten)
Voorbeelden:
- paaltjes
- kastjes
- camera’s
- bomen (alleen stam)

### Methode:
- clustering
- objecttype op basis van:  
  - hoogte  
  - footprint  
  - intensiteit  
  - verticaliteit  

Later:
- binaire classificatie → multi-class object classifier

---

## 7.9 QC & QA in objectextractie
Elke objectcategorie krijgt eigen QA-checks:

- markering: breedtecorrectheid, continuïteit, op wegvlak  
- putdeksels: diameter, hoogte, cirkel-fit residu  
- masten: cilinderresidu, verticaliteit  
- borden: vlak-fitting, hoogte, oriëntatie  
- vangrail: polyline-stabiliteit  
- trottoirband: consistent hoogteprofiel  
- bermobjecten: footprintvalidatie  

**Output:**  
`qa/object_checks/*.json`

Tiles met extreme afwijkingen gaan naar:  
- `object_review/`
- of worden in de portal als *Bad* gemarkeerd.

---

## 7.10 IMBOR-annotatie
Voor elk object voegen we IMBOR-velden toe:

- typecode  
- omschrijving  
- geometrie  
- afmetingen  
- posities  
- metadata  
- classificatiezekerheid  

**Output:**  
`objects_imbor.gpkg`  
`objects_imbor.csv`

---

## 7.11 Resultaat van Stap 7
Na deze fase heb je:

- alle primaire wegobjecten in 3D gereconstrueerd  
- correct geclassificeerde geometrieën  
- volledige IMBOR-attributie  
- QA-rapporten per objecttype  
- objecten klaar voor koppeling aan de middenlijn  

Dit vormt de input voor **Sectie 8: Wegmodel Generator v1**.

---

# 🗺️ 8. Wegmodel Generator v1 (CROW/IMBOR-ready)
*Doel: alle 3D-objecten, markeringen en geometrieën combineren tot een consistent Nederlands wegmodel — volledig automatisch opgebouwd en exporteerbaar.*

Met Sectie 7 zijn alle objecten geometrisch en semantisch afgeleid.  
In deze fase construeren we hieruit een **gestructureerd wegmodel**:

- Middenlijn  
- Rijstroken  
- Markering 96b  
- Wegprofiel (hoogte & dwarshelling)  
- Berm / talud  
- Positie van IMBOR-objecten  
- Optioneel: export naar Lanelet2 / OpenDRIVE / GeoPackage  

---

## 8.1 Koppelen van objecten aan de weg-as
De weg-as is de ruggengraat van het model.

### Input:
- as-lijn (geleverd of automatisch afgeleid)
- objectpunten (markering, masten, borden, putdeksels, vangrail, trottoirband)

### Methode:
1. Projecteer objectcentroid → dichtstbijzijnde punt op as  
2. Bepaal:
   - **s** (kettingage / chainage)  
   - **d** (dwarsoffset)  
   - **z** (hoogtewaarde t.o.v. NAP)  
3. Controleer oriëntatie (voor borden, lijnen)

**Output:**  
`objects_aligned.csv` met IMBOR-velden + (s, d, z) per object.

---

## 8.2 Reconstructie van rijstrookgeometrie
Met de BEV-markeringen en hun polyline-geometrieën reconstrueren we:

- rijstrookindeling  
- middenlijn & kantlijnen  
- verdrijvingsvlakken  
- invoeg-/uitvoegstroken  
- weefvakken  

### Methode:
1. groepeer markering per segment:  
   - asmarkering  
   - kantmarkering  
   - verdrijvingsvlak  
2. genereer rijstrookpolygons via parallel-offsets  
3. bepaal rijstrookbreedte per s  
4. detecteer splitsingen en samenvoegingen

**Output:**
lanes/
laneID_polygon.gpkg
laneID_centerline.gpkg

---

## 8.3 Markering volgens CROW 96b
De markeringstypen uit Sectie 7 worden nu:

- gevalideerd  
- geclassificeerd  
- aan rijstroken gekoppeld  
- voorzien van CROW 96b parameters:

  - type (doorgetrokken, onderbroken, waarschuwingsmarkering)
  - lengtes (markering + tussenruimte)
  - breedtes (100 / 150 / 200 mm)
  - patroon (ev. 3:1 / 1:1 / verdrijvingspunt)

### Output:
`crow_markering.gpkg` — volledige 96b markup.

---

## 8.4 Wegprofielgeneratie (hoogte & dwarshelling)
De 3D punten van het wegoppervlak worden gebruikt om:

- hoogteprofielen  
- dwarsprofielen  
- superelevatie  
- klaverblad-overgangen  
- taludvormen  

te reconstrueren.

### Methode:
- sampling per 0.5 m op s-as  
- regressie op dwarsprofielen  
- detectie van dwarshelling overgangspunten  
- verwijder outliers via RANSAC plane fitting

**Output:**
profiles/
height_profile.csv
slope_profile.csv
cross_sections.gpkg

---

## 8.5 Bermmodel & talud
Uit de gronddetectie + BEV-intensiteit + 3D profielpunten:

### Output:
- bermlijnen  
- talud-einden  
- hoogte-overgangen  
- objectposities (kastjes, palen)

---

## 8.6 Objecten plaatsen in wegmodel (IMBOR integratie)
Alle objecten uit Sectie 7 worden aan het wegmodel gekoppeld:

- masten langs de kantlijn  
- borden langs rijstrook of berm  
- putdeksels exact op wegvlak  
- vangrail uitgelijnd op rijrichting  
- bermobjecten gekoppeld aan profiel

**IMBOR attributen toegevoegd:**
- IMBOR ID  
- objecttype  
- geometrie  
- materiaal (optioneel)  
- ligging (s,d,z)

**Output:**  
`wegmodel_imbor.gpkg`

---

## 8.7 Integratie van QA
Het wegmodel krijgt QA-tags:

- `discontinuity_lane`  
- `unexpected_width_change`  
- `marking_mismatch`  
- `object_height_warning`  
- `as_alignment_uncertain`  

Deze flags worden later weergegeven in de portal.

---

## 8.8 Exportformaten (GIS & Simulatie)
We exporteren het volledige wegmodel naar:

### GeoPackage
- lane centerlines  
- lane polygons  
- markering  
- IMBOR-objecten  
- profielen  

### Lanelet2
- lanelets  
- relations  
- regulatory elements  

### OpenDRIVE (optioneel)
- road geometry  
- lane sections  
- superelevation  
- horizontal alignment  

### GeoJSON voor snelle kaartvisualisatie

---

## 8.9 Resultaat van Stap 8
Na deze fase krijg je:

- een compleet **automatisch gereconstrueerd wegmodel**  
- volledig gebaseerd op echte Riegl-data  
- inclusief CROW-markering  
- inclusief IMBOR-objecten  
- hoogte/dwarsprofielen  
- lane geometrie  
- kantlijnen & verdrijvingsvlakken  
- export naar GeoPackage / Lanelet2 / OpenDRIVE  

Dit model vormt de basis voor **Sectie 9: Validatie & Kwaliteitscontrole**.

---

# 📊 9. Validatie & Kwaliteitscontrole (End-to-End QA)
*Doel: een volledige kwaliteitscontrole van ruwe LiDAR tot eindmodel — inclusief technische QA, geometrische QA en IMBOR/CROW-validaties.*

Na Sectie 8 is de volledige keten operationeel.  
Deze sectie voert een systematische QA uit over:

- inputdata  
- tilekwaliteit  
- ML-performance  
- 3D terugprojectie  
- objectextractie  
- wegmodelconsistentie  

Alles resulteert in reproduceerbare rapporten en portal-visualisaties.

---

## 9.1 CROW-validatie van wegmarkering
Markering is in Nederland sterk gestandaardiseerd.  

### Controles:
- breedte per type (100/150/200 mm)  
- patroonlengtes (3:1, 1:1, 9:3, etc.)  
- positie t.o.v. rijstrookcentrum  
- continuïteit (gaten / inconsistenties)  
- markering vs. wegprofiel (moet op het wegvlak liggen)  
- type-identificatie (onderbroken / doorgetrokken / waarschuwing / verdrijvingsvlak)

**Output:**
qa/crow_marking_report.md
qa/crow_marking_violations.gpkg

---

## 9.2 IMBOR-validatie van objecten
Voor elk objecttype:

- Geometriecontrole (cilinder, cirkel, vlak)  
- Afmetingen binnen IMBOR-range  
- Verticaliteit (masten)  
- Reflectieconsistentie (borden)  
- Hoogte boven maaiveld  

**Checks:**
- hoogteputdeksel binnen ±3 cm van wegvlak  
- mast verticaliteit ≤ 2° helling  
- bordhoogte > 1.5 m (en afhankelijk van type)  
- vangrailhoogte binnen tolerantie  
- trottoirbandhoogte consistent over s  

**Output:**
qa/imbor_report.md
qa/imbor_violations.gpkg

---

## 9.3 Densiteit, dekking & zichtanalyse
Deze analyse bepaalt of de inputdata kwalitatief voldoende is.

### Controles:
- puntdichtheid per s  
- zichtlijnen naar objecten (occlusions)  
- onderdekking door verkeer (auto’s, vrachtwagens)  
- inconsistenties door regen/glare  

**Output:**
qa/density_heatmap.png
qa/coverage_report.md

---

## 9.4 Inference QA (ML performance)
Voor elke modelrun:

- per-klasse IoU (BEV)  
- per-object precision/recall (camera)  
- hardest false positives  
- hardest false negatives  
- uncertainty maps  
- tiles met afwijkingen automatisch gemarkeerd  

**Output:**
qa/ml_performance.html
qa/ml_confusion_matrix.png
qa/bad_inference_tiles.json

---

## 9.5 3D terugprojectie-validatie
Controleren of 2D→3D terugprojectie klopt.

### Checks:
- BEV-maskers geven punten in juiste clusters  
- camera-boxen projecteren naar consistente dieptes  
- hoogteafwijking objectpunten < drempel  
- duplicaties in fusion gedetecteerd  
- segmentering logisch:  
  - markering → lijnen  
  - putdeksel → vlakke cirkel  
  - mast → cilinder  

**Output:**
qa/3d_projection_report.md
qa/3d_projection_issues.gpkg

---

## 9.6 Wegmodel-structuurvalidatie
Het wegmodel moet intern consistent zijn.

### Controles:
- lanes overlappen niet  
- lane transitions correct  
- kantlijnen sluiten aan  
- superelevatieprofiel glad  
- dwarsprofielen consistent  
- objecten op logische posities langs berm of kantlijn  
- markering aansluitend op lane centerlines  

**Output:**
qa/roadmodel_validation.md
qa/roadmodel_issues.gpkg

---

## 9.7 Portal-integratie voor mens-in-de-loop QA
Alle QA-resultaten kunnen worden bekeken en gecorrigeerd in de webportal:

### Portal features:
- BEV-tiles visueel checken (Good/Bad)  
- cameraview kwaliteit beoordelen  
- ML-predictions naast ground truth tonen  
- 3D clusters inspecteren  
- objecten selecteren als:  
  - correct  
  - incorrect  
  - needs review  
- hertrainingstriggers automatisch genereren  

Portal genereert:
portal_feedback.json

Deze wordt direct gebruikt in:
- Sectie 5 → retraining  
- Sectie 7 → objectreconstructie-check  
- Sectie 8 → wegmodelrepair  

---

## 9.8 End-to-End QA-rapport
Er wordt een eindrapport gebouwd dat alle controles samenvat:

- inputdata (ruwe Riegl)  
- preprocessing stabiliteit  
- QA-tile verdeling  
- annotatiekwaliteit  
- ML-statistieken  
- 3D consistentie  
- IMBOR & CROW validatie  
- wegmodelproblemen  
- verbeterpunten  

**Output:**
qa/final_end_to_end_report.pdf
qa/final_end_to_end_dashboard.html

---

## 9.9 Resultaat van Stap 9
Je beschikt nu over een volledig gevalideerde, technisch en semantisch correcte keten:

- Validatie volgens CROW + IMBOR  
- Data-dekking + densiteit controles  
- ML performance analyse  
- 3D en wegmodelconsistentie  
- Portalgestuurde correctieloop  
- Eindrapport voor engineering, asset management of inspectie  

De pipeline is hiermee gereed voor **Stap 10: Geautomatiseerde Wegmodellering** en toekomstige uitbreidingen.

---

# 🏁 10. Einddoel — Volledige Automatische Wegmodellering (Productie-klaar)
*Doel: een volledig geautomatiseerde keten die ruwe mobiele LiDAR omzet in een correct, compleet en gevalideerd Nederlands wegmodel, inclusief objecten, markering, profielen en IMBOR-attributie.*

Dit is de ultieme state — een systeem dat zonder handmatige tussenstappen:

1. ruwe Riegl MMS-data inleest  
2. stabiliseert, filtert, normaliseert en QA-labelt  
3. tiles en camera views maakt  
4. ML-modellen toepast  
5. 2D resultaten terugprojecteert naar 3D  
6. objecten reconstrueert  
7. alles koppelt aan een weg-as  
8. rijstroken en geometrie reconstrueert  
9. IMBOR/CROW-conform wegmodel exporteert  
10. volledige QA-rapportage en portal-gestuurde correctielus uitvoert  

Het systeem vormt dan een reproduceerbare, schaalbare productieketen.

---

## 10.1 End-to-end automatische pipeline
De pipeline draait in Docker, GPU-accelerated, en verwerkt datasets van enkele kilometers tot volledige wegen.

### De volledige keten:
- **Input:**  
  - LAZ (ruw, meerdere swaths, ongefilterd)  
  - Trajectory (GNSS/IMU)  
  - Middenas (geleverd of afgeleid)

- **Automatische stappen:**  
  - Preprocessing (filtering, ground, intensiteit, densiteit)  
  - QA-tagging  
  - BEV 6-kanaals en 360° cameraviews  
  - ML-inference (BEV segmentatie + camera detectie)  
  - BEV → 3D & camera → 3D  
  - Objectextractie en IMBOR-classificatie  
  - Reconstructie van rijstroken & markering (CROW)  
  - Wegprofiel- en taludmodellering  
  - Wegmodel-samenstelling  
  - QA-checks  
  - Export naar engineeringformaten

**Output:**  
- volledig wegmodel  
- IMBOR-objekten  
- lane geometrie  
- rijstrookindeling  
- profielen  
- CROW-96b markering  
- QA-rapporten  

---

## 10.2 Mens-in-de-loop via webportal
Hoewel de pipeline autonoom draait, blijft menselijke supervisie essentieel voor uitzonderingen.

### Portal ondersteunt:
- BEV & CAM tile inspectie  
- annotatie (2D/3D)  
- Good/Bad labeling  
- objectvalidatie (masten, borden, markeringen)  
- wegmodel QA  
- retraining triggers  
- datasetversiebeheer  
- object editing (optioneel, later)  

**Feedback wordt verwerkt in:**  
- nieuwe ML iteraties  
- correcties in objectextractie  
- verbeterde stabilisatie of preprocessing  

---

## 10.3 Versiebeheer & Reproduceerbaarheid
Elke stap van de keten is reproduceerbaar via:

- datasetversies  
- modelversies  
- configuratiebestanden  
- QA-rapporthistorie  
- exportversies van het wegmodel  

### Opslagstructuur:
datasets/
models/
roadmodels/
qa_reports/
portal_feedback/
configs/

---

## 10.4 Schaalbaarheid & performance
De pipeline moet efficiënt werken op grote volumes Riegl-data.

### Optimalisaties:
- CUDA-accelerated kernfuncties  
- parallel preprocessing  
- batch inference  
- on-demand tile regeneration  
- incremental retraining (op basis van portal feedback)  
- asset caching  

### Doel:
- **1 km MMS-data verwerken in 2–5 minuten**  
  (afhankelijk van hardware)

---

## 10.5 Engineering-ready exports
Het systeem levert professionele, direct bruikbare outputs:

### GeoPackage (GIS)
- lane centerlines  
- lane polygons  
- markering  
- talud  
- IMBOR-objecten  
- wegprofiel  

### Lanelet2 (automated driving)
- regulatory elements  
- routing graph  
- lanelets  

### OpenDRIVE (simulatie)
- horizontale & verticale geometrie  
- superelevation  
- lane sections  
- markering & objecten  

### CSV / JSON (engineering workflows)
- objectlist  
- profielen  
- alignment data  

---

## 10.6 Cyclische verbetering via retraining-loop
Het systeem wordt beter naarmate het meer data ziet.

### Automatische verbetercyclus:
1. nieuwe dataset verwerken  
2. portal inspecteert Good/Bad tiles  
3. systeem hertraint modellen  
4. update pipeline-modules  
5. verbeterd wegmodel genereren  
6. QA bevestigt verbetering  

Deze cyclus maakt de pipeline zelflerend.

---

## 10.7 Einddoel bereikt
Wanneer deze fase operationeel is, kan het systeem:

- zonder handmatige tussenkomst  
- van 100% ruwe LiDAR  
- een 100% gevalideerd wegmodel produceren  
- conform Nederlandse IMBOR- en CROW-standaarden  
- met volledige QA en reproduceerbaarheid  
- en met een mens-in-de-loop portal voor uitzonderingen  

Dit is de definitieve vorm van de **geautomatiseerde LiDAR Wegmodellering Pipeline**.
