"""
PHASE 1 — DATA INSPECTION AND CLEANING (ILLINOIS TEST RUN)
============================================================

Purpose
-------
Prepare the Illinois highway, railway, and county-centroid datasets
for the subsequent network-construction phases.

SCOPE NOTE: the three input files are already Illinois-scoped source
extracts (they are named/structured as such), so -- unlike the
earlier California test run of this script -- NO additional
state/FIPS scoping filter is applied here. If these inputs turn out
to also contain out-of-state slivers, add an explicit Illinois filter
the same way the earlier California filter worked (state name/FIPS
column, hard-fail on zero records).

Datasets (all .gpkg)
---------------------
1. HIGHWAY   -- layer "HWY2025"                       (LineString Z)
2. RAILWAY   -- layer "illonois_railway__stracnet"     (LineString)
3. COUNTY CENTROIDS -- layer "illinois_county_centroids" (Point)

Phase 1 is ONLY responsible for:
    1. Loading raw datasets
    2. Inspecting datasets
    3. Normalizing attribute names
    4. Standardizing CRS to EPSG:4326
    5. Removing missing/empty geometries
    6. Repairing invalid geometries where possible
    7. Removing exact duplicate records
    8. Applying methodology-defined attribute cleaning rules
       (including the highway functional-class filter, see below)
    9. Validating the cleaned datasets
   10. Saving cleaned datasets (.geojson) and a quality report

Highway functional-class (FC) filter
-------------------------------------
Per project request, the HIGHWAY dataset is restricted to FC values
1-5 (Interstate / Principal Arterial / Minor Arterial / Major
Collector / Minor Collector). FC 6 (Minor Collector - Rural
reclass'd) and FC 7 (Local) are dropped, since local-access roads are
not part of the strategic network of interest.

Important scope boundary
------------------------
This script does NOT:
    - reproject to EPSG:5070
    - calculate rail lengths
    - build transportation graphs
    - split/repair network topology
    - snap facilities to graph nodes
    - deduplicate overlapping physical installations
    - construct OD pairs
    - simulate disruptions

Those operations belong to later phases.

Methodology Rules
------------------

HIGHWAY
-------
    - Keep only FC in {1, 2, 3, 4, 5}; drop FC in {6, 7}
    - Missing/non-positive speed (SP_LIM) -> 55 mph
    - Preserve the source segment-length attribute (SEG_LENGTH)
    - Missing source road length -> HARD FAILURE
    - Zero/negative source road length -> HARD FAILURE
    - Do not invent road length from geometry

RAILWAY
-------
    - Inspect STRACNET classification values
    - Keep explicit S/C classifications
    - No fragile substring matching
    - Zero S/C matches -> HARD FAILURE
    - Speed -> 49 mph
    - Source MILES/KM are preserved when present (not hard-required,
      since the original nationwide methodology does not mandate a
      rail length field in Phase 1)

COUNTY CENTROIDS
----------------
    - Keep the cleaned source records
    - Missing/empty geometries are removed
    - Non-point geometries are converted to representative points
      (none expected -- the source layer is already Point)
    - Stable source record identifiers are added after cleaning

GEOMETRY
--------
    - Missing/empty geometries are removed
    - Invalid geometries are repaired where possible
    - Remaining invalid geometries -> HARD FAILURE
    - Remaining empty/missing geometries -> HARD FAILURE

CRS
---
    - Phase 1 uses EPSG:4326
    - EPSG:5070 is NOT used here
    - Projection belongs to graph construction
    - NOTE: the county-centroid source layer arrives in EPSG:4269
      (NAD83) and is reprojected to EPSG:4326 automatically.

Input
-----
data/raw/
    illinois_highway.gpkg           (layer: HWY2025)
    illinois_railway_stracnet.gpkg  (layer: illonois_railway__stracnet)
    illinois_county_centroids.gpkg  (layer: illinois_county_centroids)

>>> If your actual filenames differ, edit the FILES dict below. <<<

Output
------
data/cleaned/
    highway_clean.geojson
    railway_clean.geojson
    county_centroids_clean.geojson

data/reports/
    phase1_cleaning_report.json

Progress bar
------------
A tqdm progress bar tracks the major pipeline stages (load / filter /
clean / validate / save, per dataset) so you can see the script is
alive during the ~589k-row highway load. Install with:
    pip install tqdm
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
REPORT_DIR = BASE_DIR / "data" / "reports"


# ============================================================
# PHASE 1 CRS
# ============================================================

SOURCE_CRS = "EPSG:4326"


# ============================================================
# METHODOLOGY SPEEDS
# ============================================================

HIGHWAY_DEFAULT_SPEED_MPH = 55.0
RAILWAY_SPEED_MPH = 49.0


# ============================================================
# HIGHWAY FUNCTIONAL CLASS FILTER
# ============================================================

HIGHWAY_ALLOWED_FC = {1, 2, 3, 4}
HIGHWAY_EXCLUDED_FC = {5, 6, 7}


# ============================================================
# INPUT FILES / LAYERS
# ============================================================
#
# Each source is a .gpkg. GeoPackages can hold multiple layers, so we
# specify layer names explicitly rather than relying on the default
# (first) layer.

FILES = {
    "highway": RAW_DIR / "illonois_highway.gpkg",
    "railway": RAW_DIR / "illonois_railway.gpkg",
    "county_centroids": RAW_DIR / "illinois_county_centroids.gpkg",
}

LAYERS = {
    "highway": "HWY2025",
    "railway": "illonois_railway__stracnet",
    "county_centroids": "illinois_county_centroids",
}

OUTPUT_FILENAMES = {
    "highway": "highway_clean.geojson",
    "railway": "railway_clean.geojson",
    "county_centroids": "county_centroids_clean.geojson",
}


# ============================================================
# 2. DIRECTORIES
# ============================================================

CLEANED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("phase1")


# ============================================================
# 4. COLUMN NORMALIZATION
# ============================================================

def clean_column_names(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Normalize attribute column names.
    """

    gdf = gdf.copy()

    geometry_column = gdf.geometry.name

    rename_map = {}

    for column in gdf.columns:

        if column == geometry_column:
            continue

        new_name = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        rename_map[column] = new_name

    return gdf.rename(
        columns=rename_map
    )


# ============================================================
# 5. GEOMETRY CLEANING
# ============================================================

def remove_empty_geometries(
    gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, int]:
    """
    Remove records with missing or empty geometries.
    """

    before = len(gdf)

    valid_mask = (
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    )

    cleaned = gdf.loc[
        valid_mask
    ].copy()

    removed = before - len(cleaned)

    return cleaned, removed


def repair_invalid_geometries(
    gdf: gpd.GeoDataFrame,
    dataset_name: str,
) -> tuple[gpd.GeoDataFrame, int, int]:
    """
    Attempt to repair invalid geometries using make_valid().

    Returns
    -------
    cleaned_gdf
    invalid_before
    invalid_after
    """

    gdf = gdf.copy()

    invalid_before = int(
        (~gdf.geometry.is_valid).sum()
    )

    if invalid_before == 0:

        return (
            gdf,
            0,
            0,
        )

    logger.warning(
        "%s: %d invalid geometries detected.",
        dataset_name,
        invalid_before,
    )

    logger.info(
        "%s: attempting geometry repair.",
        dataset_name,
    )

    try:

        invalid_mask = (
            ~gdf.geometry.is_valid
        )

        gdf.loc[
            invalid_mask,
            "geometry",
        ] = (
            gdf.loc[
                invalid_mask,
                "geometry",
            ].make_valid()
        )

    except Exception as exc:

        logger.error(
            "%s: geometry repair failed: %s",
            dataset_name,
            exc,
        )

        raise

    invalid_after = int(
        (~gdf.geometry.is_valid).sum()
    )

    if invalid_after == 0:

        logger.info(
            "%s: geometry repair successful.",
            dataset_name,
        )

    else:

        logger.error(
            "%s: %d invalid geometries remain.",
            dataset_name,
            invalid_after,
        )

        raise ValueError(
            f"{dataset_name}: {invalid_after} invalid "
            "geometries remain after repair."
        )

    return (
        gdf,
        invalid_before,
        invalid_after,
    )


# ============================================================
# 6. DUPLICATE CLEANING
# ============================================================

def remove_exact_duplicates(
    gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, int]:
    """
    Remove exact duplicate rows.

    This is intentionally conservative. It does NOT attempt to
    identify overlapping physical facilities or topologically
    duplicate network segments.
    """

    before = len(gdf)

    cleaned = (
        gdf
        .drop_duplicates()
        .copy()
    )

    removed = before - len(cleaned)

    return (
        cleaned,
        removed,
    )


# ============================================================
# 7. CRS VALIDATION
# ============================================================

def ensure_source_crs(
    gdf: gpd.GeoDataFrame,
    dataset_name: str,
) -> gpd.GeoDataFrame:
    """
    Ensure a CRS exists and normalize to EPSG:4326.

    If the source CRS is missing, fail rather than guessing.
    """

    if gdf.crs is None:

        raise ValueError(
            f"{dataset_name}: CRS is missing. "
            "The source CRS must be verified manually. "
            "The script will not guess it."
        )

    current_crs = str(
        gdf.crs
    )

    logger.info(
        "%s: source CRS = %s",
        dataset_name,
        current_crs,
    )

    if current_crs != SOURCE_CRS:

        logger.info(
            "%s: converting %s -> %s",
            dataset_name,
            current_crs,
            SOURCE_CRS,
        )

        gdf = gdf.to_crs(
            SOURCE_CRS
        )

    return gdf


# ============================================================
# 8. LOAD + COMMON CLEANING
# ============================================================

def load_dataset(
    path: Path,
    dataset_name: str,
    layer: Optional[str] = None,
) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Load a raw dataset (optionally a specific .gpkg layer) and
    perform common Phase 1 cleaning.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"{dataset_name}: file not found:\n"
            f"{path}\n"
            "Edit the FILES dict at the top of this script if your "
            "raw .gpkg files live somewhere else or are named "
            "differently."
        )

    logger.info("")
    logger.info("=" * 75)
    logger.info(
        "LOADING %s",
        dataset_name,
    )
    logger.info("=" * 75)

    logger.info(
        "File: %s",
        path,
    )

    if layer is not None:

        logger.info(
            "Layer: %s",
            layer,
        )

    gdf = gpd.read_file(
        path,
        layer=layer,
    )

    original_count = len(gdf)

    logger.info(
        "%s: raw records = %d",
        dataset_name,
        original_count,
    )

    logger.info(
        "%s: original CRS = %s",
        dataset_name,
        gdf.crs,
    )

    logger.info(
        "%s: original columns = %s",
        dataset_name,
        list(gdf.columns),
    )

    # --------------------------------------------------------
    # Column names
    # --------------------------------------------------------

    gdf = clean_column_names(
        gdf
    )

    # --------------------------------------------------------
    # CRS
    # --------------------------------------------------------

    gdf = ensure_source_crs(
        gdf,
        dataset_name,
    )

    # --------------------------------------------------------
    # Empty geometries
    # --------------------------------------------------------

    (
        gdf,
        empty_removed,
    ) = remove_empty_geometries(
        gdf
    )

    if empty_removed:

        logger.warning(
            "%s: removed %d missing/empty geometries.",
            dataset_name,
            empty_removed,
        )

    # --------------------------------------------------------
    # Invalid geometries
    # --------------------------------------------------------

    (
        gdf,
        invalid_before,
        invalid_after,
    ) = repair_invalid_geometries(
        gdf,
        dataset_name,
    )

    # --------------------------------------------------------
    # Exact duplicates
    # --------------------------------------------------------

    (
        gdf,
        duplicates_removed,
    ) = remove_exact_duplicates(
        gdf
    )

    if duplicates_removed:

        logger.info(
            "%s: removed %d exact duplicate records.",
            dataset_name,
            duplicates_removed,
        )

    report = {
        "dataset": dataset_name,
        "input_file": str(path),
        "layer": layer,
        "original_records": original_count,
        "records_after_common_cleaning": len(gdf),
        "empty_geometries_removed": empty_removed,
        "invalid_geometries_before_repair": invalid_before,
        "invalid_geometries_after_repair": invalid_after,
        "exact_duplicates_removed": duplicates_removed,
        "crs": str(gdf.crs),
    }

    return (
        gdf,
        report,
    )


# ============================================================
# 9. COLUMN FINDER
# ============================================================

def find_column(
    gdf: gpd.GeoDataFrame,
    candidates: list[str],
    required: bool = True,
) -> Optional[str]:
    """
    Find a column from a list of possible names.
    """

    available = {
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in gdf.columns
    }

    for candidate in candidates:

        normalized_candidate = (
            str(candidate)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if normalized_candidate in available:

            for column in gdf.columns:

                normalized_column = (
                    str(column)
                    .strip()
                    .lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                )

                if normalized_column == normalized_candidate:

                    return column

    if required:

        raise KeyError(
            "Required column was not found.\n"
            f"Candidates: {candidates}\n"
            f"Available columns: {list(gdf.columns)}"
        )

    return None


# ============================================================
# 10. STABLE SOURCE IDENTIFIERS
# ============================================================

def add_stable_record_id(
    gdf: gpd.GeoDataFrame,
    dataset_key: str,
) -> gpd.GeoDataFrame:
    """
    Add a stable Phase-1 record identifier.

    Existing source identifiers are preserved. The generated
    phase1_record_id is only an audit/tracing identifier and is
    deliberately added AFTER exact-duplicate removal.
    """

    gdf = gdf.copy()

    if "phase1_record_id" in gdf.columns:

        raise ValueError(
            f"{dataset_key}: phase1_record_id already exists."
        )

    prefix = dataset_key.upper()

    gdf["phase1_record_id"] = [
        f"{prefix}_{i:07d}"
        for i in range(1, len(gdf) + 1)
    ]

    return gdf


# ============================================================
# 11. HIGHWAY FUNCTIONAL-CLASS FILTER
# ============================================================

def filter_highway_functional_class(
    gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Restrict the highway dataset to functional classes 1-5.

    FC 6 (Minor Collector, rural-reclassified) and FC 7 (Local) are
    dropped -- these are local-access roads, not part of the
    strategic network.

    HARD FAILURE if:
      - the FC column can't be found
      - the filter leaves zero records
    """

    fc_column = find_column(
        gdf,
        [
            "fc",
            "func_class",
            "functional_class",
            "functclass",
        ],
        required=True,
    )

    before = len(gdf)

    fc_numeric = pd.to_numeric(
        gdf[fc_column],
        errors="coerce",
    )

    unmatched_fc_count = int(
        fc_numeric.isna().sum()
    )

    if unmatched_fc_count:

        logger.warning(
            "HIGHWAY: %d records have a non-numeric/missing FC "
            "value and will be dropped by the FC filter.",
            unmatched_fc_count,
        )

    unique_fc_values = sorted(
        fc_numeric
        .dropna()
        .unique()
        .tolist()
    )

    logger.info(
        "HIGHWAY: unique FC values in source = %s",
        unique_fc_values,
    )

    keep_mask = fc_numeric.isin(
        HIGHWAY_ALLOWED_FC
    )

    filtered = gdf.loc[
        keep_mask
    ].copy()

    dropped_fc6_7 = int(
        fc_numeric.isin(HIGHWAY_EXCLUDED_FC).sum()
    )

    logger.info(
        "HIGHWAY: FC filter (column=%s) kept %d/%d records "
        "(dropped %d records with FC in %s, %d records with "
        "unmatched/missing FC).",
        fc_column,
        len(filtered),
        before,
        dropped_fc6_7,
        sorted(HIGHWAY_EXCLUDED_FC),
        unmatched_fc_count,
    )

    if len(filtered) == 0:

        raise ValueError(
            "HIGHWAY: FC filter left 0 records. Investigate the "
            "FC column values before trusting an empty result."
        )

    return filtered, {
        "dataset": "HIGHWAY",
        "fc_column": fc_column,
        "unique_source_fc_values": unique_fc_values,
        "allowed_fc": sorted(HIGHWAY_ALLOWED_FC),
        "excluded_fc": sorted(HIGHWAY_EXCLUDED_FC),
        "records_before_fc_filter": before,
        "records_after_fc_filter": len(filtered),
        "records_dropped_fc6_7": dropped_fc6_7,
        "records_dropped_unmatched_fc": unmatched_fc_count,
    }


# ============================================================
# 12. HIGHWAY CLEANING
# ============================================================

def clean_highway(
    gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Apply Phase 1 highway cleaning rules.

    Rules
    -----
    Missing speed -> 55 mph
    Non-positive speed -> 55 mph
    Preserve source segment length
    Missing/zero/negative source segment length -> HARD FAILURE
    """

    logger.info("")
    logger.info("=" * 75)
    logger.info("HIGHWAY CLEANING")
    logger.info("=" * 75)

    gdf = gdf.copy()

    # --------------------------------------------------------
    # Find speed field
    # --------------------------------------------------------

    speed_column = find_column(
        gdf,
        [
            "sp_lim",
            "speed_mph",
            "speed",
            "speed_limit",
            "speedlimit",
            "speed_limi",
            "speed_lmt",
            "posted_spd",
            "mph",
        ],
        required=False,
    )

    # --------------------------------------------------------
    # Find source segment-length field
    # --------------------------------------------------------

    length_column = find_column(
        gdf,
        [
            "seg_length",
            "road_length",
            "length",
            "road_len",
            "shape_leng",
            "shape_length",
            "miles",
            "connmiles",
            "distance_miles",
            "road_miles",
        ],
        required=False,
    )

    # ========================================================
    # SPEED
    # ========================================================

    if speed_column is None:

        logger.warning(
            "HIGHWAY: speed column not found."
        )

        logger.warning(
            "HIGHWAY: assigning 55 mph to all records."
        )

        gdf["speed_mph"] = (
            HIGHWAY_DEFAULT_SPEED_MPH
        )

        missing_speed_count = len(gdf)
        non_positive_speed_count = 0

    else:

        gdf["speed_mph"] = pd.to_numeric(
            gdf[speed_column],
            errors="coerce",
        )

        missing_mask = (
            gdf["speed_mph"].isna()
        )

        missing_speed_count = int(
            missing_mask.sum()
        )

        # Source zero/non-positive values are treated as missing
        # speed-limit information for this dataset.
        non_positive_mask = (
            gdf["speed_mph"] <= 0
        )

        non_positive_speed_count = int(
            non_positive_mask.sum()
        )

        missing_or_invalid_mask = (
            missing_mask | non_positive_mask
        )

        if missing_speed_count or non_positive_speed_count:

            logger.info(
                "HIGHWAY: fixing %d missing or invalid speed values "
                "(including source zeros used as missing speed limits).",
                int(missing_or_invalid_mask.sum()),
            )

            gdf.loc[
                missing_or_invalid_mask,
                "speed_mph",
            ] = (
                HIGHWAY_DEFAULT_SPEED_MPH
            )

    # --------------------------------------------------------
    # Explicit post-cleaning speed validation
    # --------------------------------------------------------

    remaining_invalid_speed = int(
        (
            gdf["speed_mph"].isna()
            | (gdf["speed_mph"] <= 0)
        ).sum()
    )

    if remaining_invalid_speed:

        raise ValueError(
            "HIGHWAY: "
            f"{remaining_invalid_speed} invalid speed values "
            "remain after cleaning."
        )

    logger.info(
        "HIGHWAY: remaining invalid speed values = 0."
    )

    # ========================================================
    # ROAD LENGTH
    # ========================================================

    missing_length_count = 0
    invalid_length_count = 0

    if length_column is None:

        # This is deliberately a HARD FAILURE. The methodology
        # requires preservation of the source road-length attribute;
        # geometry-derived length must not silently replace it.
        raise ValueError(
            "HIGHWAY: source road-length column was not found. "
            "The source road-length attribute is required; "
            "no artificial or geometry-derived road length will "
            "be created in Phase 1."
        )

    gdf["source_road_length"] = pd.to_numeric(
        gdf[length_column],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Missing length
    # --------------------------------------------------------

    missing_length_mask = (
        gdf["source_road_length"].isna()
    )

    missing_length_count = int(
        missing_length_mask.sum()
    )

    if missing_length_count:

        logger.error(
            "HIGHWAY: %d records have missing source road length.",
            missing_length_count,
        )

        raise ValueError(
            "HIGHWAY contains missing source road lengths. "
            "The source road-length attribute must be complete."
        )

    # --------------------------------------------------------
    # Zero / negative length
    # --------------------------------------------------------

    invalid_length_mask = (
        gdf["source_road_length"] <= 0
    )

    invalid_length_count = int(
        invalid_length_mask.sum()
    )

    if invalid_length_count:

        logger.error(
            "HIGHWAY: %d records have zero/negative "
            "source road length.",
            invalid_length_count,
        )

        raise ValueError(
            "HIGHWAY contains zero/negative source "
            "road lengths. Verify the raw dataset."
        )

    logger.info(
        "HIGHWAY: source road length validation PASSED."
    )

    # --------------------------------------------------------
    # Network identifier
    # --------------------------------------------------------

    gdf["network"] = "HIGHWAY"

    report = {
        "dataset": "HIGHWAY",
        "records": len(gdf),
        "speed_column": speed_column,
        "source_length_column": length_column,
        "missing_speed_fixed": missing_speed_count,
        "non_positive_speed_fixed": non_positive_speed_count,
        "remaining_invalid_speed": remaining_invalid_speed,
        "missing_source_length": missing_length_count,
        "invalid_source_length": invalid_length_count,
        "default_speed_mph": HIGHWAY_DEFAULT_SPEED_MPH,
        "source_length_preserved": True,
        "geometry_length_not_used_as_source_length": True,
    }

    return (
        gdf,
        report,
    )


# ============================================================
# 13. RAILWAY CLEANING
# ============================================================

def clean_railway(
    gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Apply Phase 1 railway (STRACNET) cleaning rules.

    Rules
    -----
    1. Retain all railway records without classification filtering.
    2. Assign 49 mph to all railway segments.
    3. Preserve source MILES/KM when present (not hard-required).
    """

    logger.info("")
    logger.info("=" * 75)
    logger.info("RAILWAY CLEANING")
    logger.info("=" * 75)

    gdf = gdf.copy()

    records_before = len(gdf)

    logger.info(
        "RAILWAY: retaining all %d records; no classification filter applied.",
        len(gdf),
    )

    gdf["speed_mph"] = (
        RAILWAY_SPEED_MPH
    )

    remaining_invalid_speed = int(
        (
            gdf["speed_mph"].isna()
            | (gdf["speed_mph"] <= 0)
        ).sum()
    )

    if remaining_invalid_speed:

        raise ValueError(
            "RAILWAY: invalid speed assignment detected."
        )

    gdf["network"] = "RAILWAY"

    # --------------------------------------------------------
    # Optional: preserve source MILES / KM (not hard-required)
    # --------------------------------------------------------

    miles_column = find_column(
        gdf,
        ["miles"],
        required=False,
    )

    km_column = find_column(
        gdf,
        ["km"],
        required=False,
    )

    if miles_column is not None:

        gdf["source_length_miles"] = pd.to_numeric(
            gdf[miles_column],
            errors="coerce",
        )

    if km_column is not None:

        gdf["source_length_km"] = pd.to_numeric(
            gdf[km_column],
            errors="coerce",
        )

    report = {
        "dataset": "RAILWAY",
        "records_before_cleaning": records_before,
        "records_after_cleaning": len(gdf),
        "classification_filter_applied": False,
        "speed_mph": RAILWAY_SPEED_MPH,
        "remaining_invalid_speed": remaining_invalid_speed,
        "source_miles_column": miles_column,
        "source_km_column": km_column,
    }

    return (
        gdf,
        report,
    )


# ============================================================
# 14. FACILITY (COUNTY CENTROID) CLEANING
# ============================================================

def clean_facility_dataset(
    gdf: gpd.GeoDataFrame,
    dataset_name: str,
) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Clean a point facility dataset (here: county centroids).

    Non-point geometries are converted to representative points
    when necessary (none expected for this source layer).
    """

    logger.info("")
    logger.info("=" * 75)
    logger.info(
        "%s CLEANING",
        dataset_name.upper(),
    )
    logger.info("=" * 75)

    gdf = gdf.copy()

    original_geometry_types = (
        gdf.geometry
        .geom_type
        .value_counts()
        .to_dict()
    )

    logger.info(
        "%s geometry types: %s",
        dataset_name,
        original_geometry_types,
    )

    non_point_mask = (
        gdf.geometry.geom_type != "Point"
    )

    non_point_count = int(
        non_point_mask.sum()
    )

    if non_point_count:

        logger.warning(
            "%s: %d non-point geometries detected.",
            dataset_name,
            non_point_count,
        )

        logger.warning(
            "%s: converting non-point geometries "
            "to representative points.",
            dataset_name,
        )

        gdf.loc[
            non_point_mask,
            "geometry",
        ] = (
            gdf.loc[
                non_point_mask,
                "geometry",
            ].representative_point()
        )

    remaining_non_points = int(
        (
            gdf.geometry.geom_type != "Point"
        ).sum()
    )

    if remaining_non_points:

        raise ValueError(
            f"{dataset_name}: "
            f"{remaining_non_points} non-point geometries "
            "remain after conversion."
        )

    gdf["facility_type"] = (
        dataset_name.upper()
    )

    report = {
        "dataset": dataset_name,
        "records": len(gdf),
        "original_geometry_types": (
            original_geometry_types
        ),
        "non_point_geometries_converted": (
            non_point_count
        ),
        "remaining_non_point_geometries": (
            remaining_non_points
        ),
        "phase1_final_analysis_selection": False,
    }

    return (
        gdf,
        report,
    )


# ============================================================
# 15. FINAL VALIDATION
# ============================================================

def validate_final_geometry(
    gdf: gpd.GeoDataFrame,
    dataset_name: str,
) -> dict:
    """
    Final Phase 1 geometry and CRS validation.

    Remaining missing, empty, or invalid geometries
    cause a HARD FAILURE.
    """

    missing_count = int(
        gdf.geometry.isna().sum()
    )

    empty_count = int(
        gdf.geometry.is_empty.sum()
    )

    invalid_count = int(
        (~gdf.geometry.is_valid).sum()
    )

    geometry_types = (
        gdf.geometry
        .geom_type
        .value_counts()
        .to_dict()
    )

    if missing_count:

        raise ValueError(
            f"{dataset_name}: "
            f"{missing_count} missing geometries remain."
        )

    if empty_count:

        raise ValueError(
            f"{dataset_name}: "
            f"{empty_count} empty geometries remain."
        )

    if invalid_count:

        raise ValueError(
            f"{dataset_name}: "
            f"{invalid_count} invalid geometries remain."
        )

    if gdf.crs is None:

        raise ValueError(
            f"{dataset_name}: CRS is missing."
        )

    if str(gdf.crs) != SOURCE_CRS:

        raise ValueError(
            f"{dataset_name}: expected CRS "
            f"{SOURCE_CRS}, found {gdf.crs}."
        )

    logger.info(
        "%s final validation PASSED.",
        dataset_name,
    )

    return {
        "dataset": dataset_name,
        "records": len(gdf),
        "missing_geometries": missing_count,
        "empty_geometries": empty_count,
        "invalid_geometries": invalid_count,
        "geometry_types": geometry_types,
        "crs": str(gdf.crs),
        "validation_status": "PASSED",
    }


# ============================================================
# 16. SAVE GEOJSON
# ============================================================

def save_geojson(
    gdf: gpd.GeoDataFrame,
    filename: str,
) -> Path:
    """
    Save cleaned GeoJSON.

    Raw datasets are never modified.
    """

    output_path = (
        CLEANED_DIR
        / filename
    )

    gdf.to_file(
        output_path,
        driver="GeoJSON",
    )

    logger.info(
        "Saved: %s",
        output_path,
    )

    return output_path


# ============================================================
# 17. MAIN
# ============================================================

# Named pipeline stages, purely for the progress bar. Each dataset
# gets: load, (filter,) clean, validate, save.
PIPELINE_STAGES = [
    "Highway: load",
    "Highway: FC filter",
    "Highway: clean (speed/length)",
    "Highway: validate",
    "Highway: save",
    "Railway: load",
    "Railway: clean (S/C, speed)",
    "Railway: validate",
    "Railway: save",
    "County centroids: load",
    "County centroids: clean",
    "County centroids: validate",
    "County centroids: save",
    "Write cleaning report",
]


def main():

    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "PHASE 1 — DATA INSPECTION AND CLEANING (ILLINOIS)"
    )
    logger.info("=" * 80)

    logger.info(
        "SCOPE: Illinois highway / railway / county-centroid source "
        "extracts. No additional state/FIPS scoping filter is "
        "applied (inputs are already Illinois-scoped)."
    )

    logger.info(
        "Phase 1 CRS: %s",
        SOURCE_CRS,
    )

    logger.info(
        "Highway default speed: %.1f mph",
        HIGHWAY_DEFAULT_SPEED_MPH,
    )

    logger.info(
        "Railway speed: %.1f mph",
        RAILWAY_SPEED_MPH,
    )

    logger.info(
        "Highway FC filter: keep %s, drop %s",
        sorted(HIGHWAY_ALLOWED_FC),
        sorted(HIGHWAY_EXCLUDED_FC),
    )

    logger.info(
        "EPSG:5070 is reserved for graph construction."
    )

    logger.info("=" * 80)

    reports = []

    pbar = tqdm(
        total=len(PIPELINE_STAGES),
        desc="Phase 1 pipeline",
        unit="stage",
    )

    def advance(stage_label: str):
        pbar.set_description(stage_label)
        pbar.update(1)

    # ========================================================
    # HIGHWAY
    # ========================================================

    highway, report = load_dataset(
        FILES["highway"],
        "HIGHWAY",
        layer=LAYERS["highway"],
    )
    reports.append(report)
    advance(PIPELINE_STAGES[0])

    highway, report = filter_highway_functional_class(
        highway
    )
    reports.append(report)
    advance(PIPELINE_STAGES[1])

    highway, report = clean_highway(
        highway
    )
    reports.append(report)

    highway = add_stable_record_id(
        highway,
        "highway",
    )

    reports.append({
        "dataset": "HIGHWAY",
        "phase1_record_id_column": "phase1_record_id",
        "unique_phase1_record_ids": int(
            highway["phase1_record_id"].nunique()
        ),
    })
    advance(PIPELINE_STAGES[2])

    report = validate_final_geometry(
        highway,
        "HIGHWAY",
    )
    reports.append(report)
    advance(PIPELINE_STAGES[3])

    save_geojson(
        highway,
        OUTPUT_FILENAMES["highway"],
    )
    advance(PIPELINE_STAGES[4])

    # ========================================================
    # RAILWAY
    # ========================================================

    railway, report = load_dataset(
        FILES["railway"],
        "RAILWAY",
        layer=LAYERS["railway"],
    )
    reports.append(report)
    advance(PIPELINE_STAGES[5])

    railway, report = clean_railway(
        railway
    )
    reports.append(report)

    railway = add_stable_record_id(
        railway,
        "railway",
    )

    reports.append({
        "dataset": "RAILWAY",
        "phase1_record_id_column": "phase1_record_id",
        "unique_phase1_record_ids": int(
            railway["phase1_record_id"].nunique()
        ),
    })
    advance(PIPELINE_STAGES[6])

    report = validate_final_geometry(
        railway,
        "RAILWAY",
    )
    reports.append(report)
    advance(PIPELINE_STAGES[7])

    save_geojson(
        railway,
        OUTPUT_FILENAMES["railway"],
    )
    advance(PIPELINE_STAGES[8])

    # ========================================================
    # COUNTY CENTROIDS
    # ========================================================

    county_centroids, report = load_dataset(
        FILES["county_centroids"],
        "COUNTY_CENTROIDS",
        layer=LAYERS["county_centroids"],
    )
    reports.append(report)
    advance(PIPELINE_STAGES[9])

    county_centroids, report = clean_facility_dataset(
        county_centroids,
        "COUNTY_CENTROIDS",
    )
    reports.append(report)

    county_centroids = add_stable_record_id(
        county_centroids,
        "county_centroids",
    )

    reports.append({
        "dataset": "COUNTY_CENTROIDS",
        "phase1_record_id_column": "phase1_record_id",
        "unique_phase1_record_ids": int(
            county_centroids["phase1_record_id"].nunique()
        ),
    })
    advance(PIPELINE_STAGES[10])

    report = validate_final_geometry(
        county_centroids,
        "COUNTY_CENTROIDS",
    )
    reports.append(report)
    advance(PIPELINE_STAGES[11])

    save_geojson(
        county_centroids,
        OUTPUT_FILENAMES["county_centroids"],
    )
    advance(PIPELINE_STAGES[12])

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report_path = (
        REPORT_DIR
        / "phase1_cleaning_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            reports,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    advance(PIPELINE_STAGES[13])
    pbar.close()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    logger.info("")
    logger.info("=" * 80)
    logger.info(
        "PHASE 1 COMPLETE"
    )
    logger.info("=" * 80)

    logger.info(
        "Highway records : %d",
        len(highway),
    )

    logger.info(
        "Railway records : %d",
        len(railway),
    )

    logger.info(
        "County centroid records : %d",
        len(county_centroids),
    )

    logger.info(
        "Cleaned data : %s",
        CLEANED_DIR,
    )

    logger.info(
        "Cleaning report : %s",
        report_path,
    )

    logger.info(
        "Raw datasets were not modified."
    )

    logger.info(
        "Output CRS: %s",
        SOURCE_CRS,
    )

    logger.info("=" * 80)


# ============================================================
# 18. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        logger.exception(
            "PHASE 1 FAILED: %s",
            exc,
        )

        raise