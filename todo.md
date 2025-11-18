LiDAR‑Pipeline TODO‑lijst
1. Input‑QA en lichte preprocessing

- Start bij `corrected_points.laz` (vendor-gecorrigeerd, RD/NAP, RGB + intensity).
- Schrijf `meta_scan.json` met `crs_ok`, `has_rgb`, `has_intensity`, `bbox`, `provider`.
- Filter AOI-outliers, extreme Z-outliers en zwarte/low-intensity ruis → `scan_filtered.laz` + `basic_filter_ok`.

2. Grond, intensiteit en tiling

- Run PMF/CSF ground filter; voeg `is_ground` en `z_rel_ground` toe; flag tiles met `ground_ok` / `uncertain_ground`.
- Clamp intensity naar [1,255], check RGB-kanalen; flag `intensity_ok`, `rgb_ok`.
- Snij in 20–25 m tiles en bereken QA: `density_ok`, `ground_ok`, `intensity_ok`, `crs_ok` → `pre_tiles/*.parquet`.

3. BEV‑ en camera‑weergeneratie

- Rasteriseer QA-goedgekeurde tiles naar BEV (6 kanalen) en schrijf pixel→punt-mappings.
- Genereer camera-views langs de as; gebruik QA om slechte tiles/views te skippen.

4. Annotatie en datasetbouw

- Bouw een annotatieportaal met BEV‑, camera‑ en 3D‑viewers; ondersteun “good” / “bad” / “needs review”‑labels.
- Exporteer BEV‑tiles en camera‑frames met bijbehorende maskers/labels en mapping‑bestanden.

5. Machine‑learning‑training

- Bouw dataloaders voor BEV‑segmentatie en camera‑detectie en definieer hyperparameters via YAML‑configuraties.
- Train U‑Net/SegFormer‑modellen voor BEV en YOLO/DETR‑modellen voor camera; monitor mIoU en mAP.
- Implementeer datasetdrift‑detectie en automatische retraining.

6. Inference en terugprojectie

- Voer inferentie uit op nieuwe tiles/views en filter resultaten via QA‑flags.
- Projecteer segmentaties en detecties terug naar 3D via de mapping; fuseer resultaten tot clusters (bv. HDBSCAN).

7. Objectextractie

- Classificeer clusters als markeringen, putdeksels, masten, borden of vangrails.
- Pas RANSAC toe om lijnen, cirkels, cilinders en vlakke borden te fitten.
- Stitch objecten over tegelgrenzen heen en fuseer overlappende swaths.

8. Wegmodel‑reconstructie

- Projecteer objecten op de as en bepaal dwarsprofielen.
- Reconstrueer rijstrookgeometrie volgens CROW 96b; bouw lane‑polygons en verwerk wegmarkeringen.
- Genereer hoogteprofielen per 0,5 m en exporteer wegmodellen naar IMBOR, Lanelet2 en OpenDRIVE.

9. Validatie en QA

- Rapporteer ML‑prestaties (mIoU, mAP) en produceer foutmatrices.
- Controleer CROW‑breedtes, markeringafstanden en objecthoogten; detecteer overlap en hiaten in lane‑constructies.
- Valideer IMBOR/CROW‑conformiteit en log afwijkingen.

10. Portal en automatisering

- Breid het portal uit met datasetbeheer, retraining‑triggers en exportmodules.
- Containeriseer preprocessing, training en inferentie; orkestreer via Docker Compose of Kubernetes.

11. Documentatie en configuratie

- Houd configuratiebestanden (bijv. preprocessing.yaml, train.yaml) up‑to‑date.
- Documenteer de pipeline, modelarchitectuur en gehanteerde normen.