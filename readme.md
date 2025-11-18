## LiDAR Road Modeling Pipeline

Deze repository bevat een startcodebase voor een volledige LiDAR‑gebaseerde wegmodellering‑pipeline. Hij is bedoeld als fundament voor onderzoek of productie, uitgaande van **leverancier‑gecorrigeerde** mobiele mapping‑data (bijv. Riegl MMS met GNSS/IMU‑correctie, RGB en RD/NAP‑coördinaten) die wordt doorverwerkt tot bruikbare wegmodellen en infrastructuurinventarisaties. De modules volgen een logische dataflow van gecorrigeerde puntenwolk naar finale laangeometrie.

### Top level folders

* `docker/` – Dockerfiles voor preprocessing, training en inferentie. Alleen de minimale afhankelijkheden per taak worden geïnstalleerd.
* `src/` – Alle Python‑broncode, onderverdeeld in logische subpackages:
  * `preprocessing/` – Lichte QA op gecorrigeerde LiDAR (metadata‑checks, basisfiltering), grond- en intensiteitsnormalisatie, tiling, BEV‑ en camera‑view generatie, plus QA‑flagging voor tile‑selectie. Optionele zware trajectstabilisatie blijft mogelijk voor echte raw‑exporten.
  * `mapping/` – Functies voor mapping tussen 2D‑pixels en 3D‑punten, sensor‑alignatie en multisensor‑fusie.
  * `ml/` – ML‑pipelines voor BEV‑segmentatie en camera‑detectie, inclusief dataset‑loaders, modeldefinities, trainingsloops en inferentie‑scripts.
  * `object_extraction/` – Tools om objecten zoals markering, putdeksels, masten, borden en vangrails uit 3D‑clusters te halen.
  * `roadmodel/` – Reconstructie van laangeometrie, CROW/IMBOR‑markeringen, dwarsprofielen en exportformaten.
  * `qa/` – Kwaliteitscontroles voor data en modeloutput.
  * `utils/` – Algemene hulpfuncties zoals logging, config‑loading, geodesie‑berekeningen en tiling‑helpers.
* `configs/` – YAML‑configuraties voor preprocessing en training.
* `portal/` – Placeholder voor een toekomstig portaal voor annotatie en QA.

### Gebruik van deze repository

Deze codebase biedt bouwstenen, geen kant‑en‑klare oplossing. Veel functies bevatten vereenvoudigde implementaties of TODO‑stubs waar domeinspecifieke logica moet worden ingevuld. Dankzij de consistente interfaces is uitbreiden echter rechttoe‑rechtaan. Enkele ankerpunten:

* `TrajectoryCorrector` in `preprocessing/trajectory_corrector.py` gebruikt een simpele moving average; zet hem alleen aan bij ruwe Riegl‑exporten of breid uit met Kalman filtering als onbewerkte GNSS/IMU‑trajecten binnenkomen.
* De LiDAR‑filters in `preprocessing/lidar_filter.py` bieden bereik‑ en intensiteitsfilters en een eenvoudige spike‑filter; gebruik ze als lichte cleaning bovenop vendor‑preprocessing, of breid uit met ghost removal/density equalisation voor raw‑data.
* `intensity_normalizer.py` voert range‑compensatie en K‑means pre‑classificatie uit op intensiteit en ruwheid; vervang dit door materiaalmodellen indien gewenst.
* `tiler.py` knipt de puntenwolk in vierkante tiles en berekent basis‑QA‑flags; voeg uitgebreidere metrics toe (roll/pitch‑stabiliteit, misalignments) als de use‑case dat vraagt.
* `bev_generator.py` rasteriseert LiDAR‑punten naar een beeld met hoogte‑, intensiteits‑ en dichtheidskanalen en levert een mapping van pixel → originele punt‑ID.
* `ml/bev` en `ml/cam` bevatten PyTorch‑skeletten voor U‑Net en YOLO/DETR‑achtige modellen, inclusief datasetklassen, modeldefinities, trainingsscripts en Ultralytics‑inferentie voor camera.

### Afhankelijkheden installeren

Twee Dockerfiles tonen hoe je omgevingen voor preprocessing en training/inferentie bouwt:

* `docker/Dockerfile.preprocess` installeert alleen wat nodig is voor de preprocessing‑pipeline (NumPy, Pandas, scikit‑learn, Numba, OpenCV, PyArrow).
* `docker/Dockerfile.train` installeert PyTorch, PyTorch Lightning, Ultralytics YOLO en andere ML‑dependencies; breid ze uit met CUDA of hardware‑optimalisaties naar behoefte.

### Volgende stappen

Mogelijke uitbreidingen:

* Robuustere sensor‑alignatie en boresight‑calibratie in `mapping/boresight.py`.
* PDAL of andere libraries toevoegen voor geavanceerde ground‑filters of intensiteitsnormalisatie.
* Een webportal (FastAPI + React) uitwerken onder `portal/` zoals in de roadmap beschreven.
* Exportscripts toevoegen voor Lanelet2, OpenDRIVE of IMBOR‑conforme wegmodellen.

Deze structuur moet de ontwikkeling van een complete LiDAR‑verwerkings‑ en wegmodellering‑pipeline versnellen.

### Startpunt: gecorrigeerde, gekleurde MMS (AHN6-achtig)

We veronderstellen dat de scanauto een **gecorrigeerde, gekleurde puntenwolk** levert in RD/NAP (AHN6‑kwaliteit). Stap 1 van de pipeline is daarom lichte preprocessing en harde QA, geen zwaar trajectherstel. Een optioneel “ruwe Riegl‑route” blijft beschikbaar zodra je ongecorrigeerde trajecten wilt verwerken.

1. **Inputcheck & metadata** – Lees `corrected_points.laz`, verifieer CRS, RGB en intensity en schrijf een `meta_scan.json` met `crs_ok`, `has_rgb`, `has_intensity`, `bbox` en `provider`.
2. **Lichte cleaning** – Filter extreme Z‑outliers, punten buiten de AOI en gecombineerde zwarte/low‑intensity ruis. Output: `scan_filtered.laz` + flag `basic_filter_ok`.
3. **Grond en hoogte** – Run een PMF/CSF ground‑filter, label `is_ground` en voeg `z_rel_ground` toe. Flag tiles als `ground_ok` of `uncertain_ground`.
4. **Intensity/RGB‑check** – Clamp intensity naar [1, 255], controleer dat markering licht blijft en asfalt donker. Optionele range‑compensatie indien de leverancier dat niet doet.
5. **Tiling en QA** – Snij in 20–25 m tiles, bereken dichtheid, ground‑fractie en intensity‑histogrammen, en schrijf `pre_tiles/*.parquet` met flags (`density_ok`, `intensity_ok`, `ground_ok`, `crs_ok`).
6. **Tile‑selectie** – BEV‑ en cam‑generators gebruiken alleen QA‑goedgekeurde tiles; overige tiles gaan naar `qa_review/` of worden als hard negatives bewaard.

#### AHN6 als fallback

Totdat echte MMS‑runs beschikbaar zijn kun je bovenstaande stappen ook op AHN6‑tiles uitvoeren. Houd dezelfde bestandsstructuur aan (`/raw/laz`, `/metadata/`, `/trajectories/`) zodat je later probleemloos kunt wisselen naar echte scanauto‑exports. Trajectcorrectie kan uit blijven zolang de poses synthetisch of vendor‑gecorrigeerd zijn.
