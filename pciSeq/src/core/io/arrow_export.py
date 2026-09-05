"""The feather files the viewer streams, sharded by plane."""

import json
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path
from typing import Iterator, List, Tuple

logger = logging.getLogger(__name__)



def parse_coords(cell: str) -> List[Tuple[float, float]]:
    """Parse a JSON-like coords string '[[x,y], ...]' into a list of (x,y)."""
    if pd.isna(cell):
        return []
    try:
        arr = json.loads(cell)
        # Expect list of [x, y]
        out: List[Tuple[float, float]] = []
        for pair in arr:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            x, y = pair
            out.append((float(x), float(y)))
        return out
    except Exception:
        return []



def validate_df_structure(df):
    """Validate that dataframe has required columns: plane_id, label, coords"""
    required_columns = {"plane_id", "label", "coords"}

    try:
        # check columns
        actual_columns = set(df.columns)

        missing_columns = required_columns - actual_columns
        if missing_columns:
            raise ValueError(f"missing required columns: {missing_columns}")

        # Check dataframe has any data rows
        if df.empty:
            raise ValueError(f"df has no data rows")

        return True

    except Exception as e:
        raise SystemExit(f"validation failed: {e}")



def _dense_planes(df_in: pd.DataFrame, num_planes: int) -> Iterator[Tuple[int, pd.DataFrame]]:
    """
    Yield (plane_id, df) for every plane in [0, num_planes), in plane order. Planes with
    no spots yield an empty slice of df_in, so they still get a shard with the same schema.
    Walks a sorted groupby once, so only one plane's sub-frame is alive at a time.
    """
    empty = df_in.iloc[0:0]
    next_plane = 0
    for key, df in df_in.groupby("plane_id", sort=True):
        plane_id = int(key)
        while next_plane < plane_id:
            yield next_plane, empty
            next_plane += 1
        yield plane_id, df
        next_plane = plane_id + 1
    while next_plane < num_planes:
        yield next_plane, empty
        next_plane += 1



def _spots_arrow_table(df: pd.DataFrame) -> pa.Table:
    """Build the spots Arrow table for one plane. An empty slice gives the same schema."""
    # Create arrays with exact schema matching working converter
    # Column order: ['x', 'y', 'z', 'plane_id', 'spot_id', 'gene_id', 'neighbour_array', 'neighbour_prob', 'omp_score', 'omp_intensity']
    arrays = {}

    # Required columns with exact types from working converter
    if "x" in df.columns:
        arrays["x"] = pa.array(df["x"].astype("float32"))
    if "y" in df.columns:
        arrays["y"] = pa.array(df["y"].astype("float32"))
    if "z" in df.columns:
        arrays["z"] = pa.array(df["z"].astype("float32"))
    if "plane_id" in df.columns:
        arrays["plane_id"] = pa.array(df["plane_id"].astype("uint16"))
    if "spot_id" in df.columns:
        arrays["spot_id"] = pa.array(df["spot_id"].astype("uint32"))
    if "gene_id" in df.columns:
        arrays["gene_id"] = pa.array(df["gene_id"].astype("uint32"))

    # List columns
    if "neighbour_array" in df.columns:
        arrays["neighbour_array"] = pa.array(df["neighbour_array"].tolist(), type=pa.list_(pa.int32()))
    if "neighbour_prob" in df.columns:
        arrays["neighbour_prob"] = pa.array(df["neighbour_prob"].tolist(), type=pa.list_(pa.float32()))

    # Optional OMP columns
    if "omp_score" in df.columns:
        arrays["omp_score"] = pa.array(df["omp_score"].astype("float32"))
    if "omp_intensity" in df.columns:
        arrays["omp_intensity"] = pa.array(df["omp_intensity"].astype("float32"))

    # Hard misread flag (0 or 1)
    if "is_hard_misread" in df.columns:
        arrays["is_hard_misread"] = pa.array(df["is_hard_misread"].astype("uint8"))

    # NOTE: gene_name and neighbour columns are excluded to match working converter
    return pa.table(arrays)



def geneData_to_arrow(df_in: pd.DataFrame, out_dir: str = None, num_planes: int = None) -> None:
    """
    Convert spots DataFrame into one Arrow Feather file per plane.

    Args:
        df_in: Spots DataFrame from spots_summary(). Must have a 'plane_id' column.
        out_dir: The root directory to save the output 'arrow_spots' folder to.
        num_planes: Total number of planes. Every plane in [0, num_planes) gets a file
                    on disk and an entry in the manifest, empty planes included. Falls
                    back to the highest plane_id in the data when not given.
    """
    out_dir = Path(out_dir) / "viewer_data" / 'arrow_spots'
    out_dir.mkdir(parents=True, exist_ok=True)

    if "plane_id" not in df_in.columns:
        raise ValueError(
            f"geneData_to_arrow: Missing required column 'plane_id'. "
            f"Cannot shard spots by plane. Check spots_summary() output."
        )

    # gene dictionary (id -> name) from the full dataset
    gene_dict_data = dict(zip(df_in["gene_id"], df_in["gene_name"]))

    # every spot must land on a plane: NaN, negative or fractional plane_ids have nowhere to go
    plane_ids = pd.to_numeric(df_in["plane_id"], errors="coerce")
    bad = plane_ids.isna() | (plane_ids < 0) | (plane_ids % 1 != 0)
    if bad.any():
        # raise before touching the output dir, so a failed run leaves the previous
        # shards and manifest consistent instead of deleting files the manifest points at
        raise ValueError(
            f"geneData_to_arrow: {int(bad.sum())} spots have a plane_id that is NaN, negative or "
            f"non-integer. They cannot be assigned to a plane shard. Check spots_summary() output."
        )
    if not pd.api.types.is_integer_dtype(df_in["plane_id"]):
        # normalise, so groupby sorts numerically even if the column arrived as float or string
        # (safe to cast: validation above already ruled out NaN, negative, fractional)
        df_in = df_in.assign(plane_id=plane_ids.astype("int64"))

    # drop shards from previous runs, eg the old row-chunked spots_shard_NNN files.
    # validation is done, so no bad input can abort after this point. the remaining
    # risk is environmental: a crash mid-write (disk full, kill) leaves the output
    # dir torn, the old manifest pointing at deleted or half-rewritten shards. that
    # window is not new, the old chunked writer overwrote shards in place with the
    # same exposure. a staging dir plus rename swap would close it if it ever matters.
    for stale in out_dir.glob("spots_*.feather"):
        stale.unlink()

    # dense plane index: never drop planes that have spots, even if num_planes is short
    max_plane = int(plane_ids.max()) + 1 if len(df_in) else 0
    if num_planes is None:
        num_planes = 0
    num_planes = max(num_planes, max_plane)

    shards = []
    total_rows = 0

    for plane_id, df in _dense_planes(df_in, num_planes):
        table = _spots_arrow_table(df)
        shard_name = f"spots_plane_{plane_id:03d}.feather"
        feather.write_feather(table, (out_dir / shard_name).as_posix(), compression='uncompressed')

        shards.append({"url": shard_name, "rows": int(len(df)), "plane": plane_id})
        total_rows += len(df)

    # Write manifest. Shards are already in plane order by construction.
    manifest = {
        "format": "arrow-feather",
        "total_rows": int(total_rows),
        "shards": shards,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Write gene dictionary (id -> name)
    (out_dir / "gene_dict.json").write_text(json.dumps(gene_dict_data, indent=2))

    # logger.info(f"Saved {total_rows} rows in {len(shards)} shards at {out_dir}")
    logger.info(f"Saved at {out_dir}")



def cellData_to_arrow(df_in: pd.DataFrame, out_dir: str = None) -> None:
    """
    Convert cell data DataFrame to Arrow Feather shards for JavaScript viewer.

    Output schema (8 columns):
      - cell_id: int32 (cell identifier)
      - X, Y, Z: float32 (cell centroid coordinates)
      - class_name: list<string> (predicted cell types, ordered by probability)
      - prob: list<float32> (classification probabilities)
      - gene_names: list<string> (detected genes, ordered by expression)
      - gene_counts: list<float32> (gene expression counts)

    Args:
        df_in: Cell data DataFrame from cells_summary()
        out_dir: Base output directory (viewer_data/arrow_cells/ will be appended)

    Raises:
        ValueError: If required source columns are missing
    """
    out_dir = Path(out_dir) / "viewer_data" / "arrow_cells"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate required source columns - fail fast if missing
    required_source_cols = ["Cell_Num", "X", "Y", "Z", "ClassName", "Prob", "Genenames", "CellGeneCount"]
    missing = [col for col in required_source_cols if col not in df_in.columns]
    if missing:
        raise ValueError(
            f"cellData_to_arrow: Missing required source columns: {missing}. "
            f"Cannot generate Arrow files for viewer. Check cells_summary() output."
        )

    # Output schema: source column -> (output name, arrow type)
    schema_map = [
        ("Cell_Num", "cell_id", pa.int32()),
        ("X", "X", pa.float32()),
        ("Y", "Y", pa.float32()),
        ("Z", "Z", pa.float32()),
        ("ClassName", "class_name", pa.list_(pa.string())),
        ("Prob", "prob", pa.list_(pa.float32())),
        ("Genenames", "gene_names", pa.list_(pa.string())),
        ("CellGeneCount", "gene_counts", pa.list_(pa.float32())),
    ]

    shards = []
    total_rows = 0
    shard_index = 0
    chunk_size = 100_000

    for start in range(0, len(df_in), chunk_size):
        df = df_in.iloc[start:start + chunk_size]
        n = len(df)

        # Build arrays in fixed column order
        arrays = {}
        for src_col, out_col, arrow_type in schema_map:
            if isinstance(arrow_type, pa.ListType):
                # List columns: convert DataFrame lists to Arrow list arrays
                arrays[out_col] = pa.array(df[src_col].tolist(), type=arrow_type)
            else:
                # Scalar columns: explicit type casting
                arrays[out_col] = pa.array(df[src_col], type=arrow_type)

        # Create table with fixed column order
        col_names = [out_col for _, out_col, _ in schema_map]
        table = pa.table({col: arrays[col] for col in col_names})

        # Write shard
        shard_name = f"cells_shard_{shard_index:03d}.feather"
        feather.write_feather(table, (out_dir / shard_name).as_posix(), compression="uncompressed")

        shards.append({"url": shard_name, "rows": int(n)})
        total_rows += n
        shard_index += 1

    # Write manifest
    manifest = {"format": "arrow-feather", "total_rows": int(total_rows), "shards": shards}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # logger.info(f"Saved {total_rows} cell records in {len(shards)} shards at {out_dir}")
    logger.info(f"Saved at {out_dir}")



def _boundaries_to_arrow(df_in: List[pd.DataFrame], out_dir: str = None) -> None:
    out_dir = Path(out_dir) / "viewer_data" / 'arrow_boundaries'
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = []
    total_polys = 0
    total_points = 0

    for boundaries in df_in:
        if boundaries.empty:
            continue

        # Vectorized operations using NumPy arrays
        coords_array = boundaries["coords"].values
        plane_ids_array = boundaries["plane_id"].values
        cell_ids_array = boundaries["cell_id"].values

        # Pre-allocate lists with known size
        num_rows = len(boundaries)
        x_lists = []
        y_lists = []
        plane_ids = np.empty(num_rows, dtype=np.uint16)
        cell_ids = np.empty(num_rows, dtype=np.int32)

        # Vectorized coordinate extraction
        valid_idx = 0
        for i in range(num_rows):
            coords = coords_array[i]
            if not coords:  # Skip empty coordinates
                continue

            # Use NumPy for faster array operations
            coords_np = np.array(coords, dtype=np.float32)
            xs = coords_np[:, 0]
            ys = coords_np[:, 1]

            if len(xs) == 0:
                continue

            x_lists.append(xs.tolist())
            y_lists.append(ys.tolist())
            plane_ids[valid_idx] = plane_ids_array[i]
            cell_ids[valid_idx] = cell_ids_array[i]
            valid_idx += 1

        # Trim arrays to actual size
        plane_ids = plane_ids[:valid_idx]
        cell_ids = cell_ids[:valid_idx]

        if valid_idx == 0:
            continue

        # Create Arrow arrays directly without intermediate pandas Series
        plane_suffix = f"{plane_ids[0]:02d}"
        arrays = {
            "x_list": pa.array(x_lists, type=pa.list_(pa.float32())),
            "y_list": pa.array(y_lists, type=pa.list_(pa.float32())),
            "plane_id": pa.array(plane_ids),
            "label": pa.array(cell_ids),
        }

        table = pa.table(arrays)
        shard_name = f"boundaries_plane_{plane_suffix}.feather"

        # Write with optimal compression settings
        feather.write_feather(
            table,
            (out_dir / shard_name).as_posix(),
            compression="uncompressed"
        )

        # Calculate stats efficiently
        polys = len(x_lists)
        pts = sum(len(xs) for xs in x_lists)  # This is still the fastest way
        total_polys += polys
        total_points += pts

        shards.append({
            "url": shard_name,
            "rows": polys,
            "plane": int(plane_ids[0])
        })
        # print(f"Wrote {shard_name}: polys={polys}, points={pts}")

    # Manifest
    manifest = {
        "format": "arrow-feather",
        "total_rows": total_polys,  # No need for int() conversion
        "total_points": total_points,
        "shards": shards,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(f"Done. Total polys: {total_polys}. Total points: {total_points}. Files: {len(shards)}. Output: {out_dir}")



def boundaries_to_arrow(dfs_in: List[pd.DataFrame], out_dir: str, compression: str = "uncompressed"):
    """
    Converts a list of DataFrames of boundary data into one Arrow Feather file per plane.

    Args:
        dfs_in: A list of DataFrames. Each DataFrame must contain data for a single plane
                and have the columns ['plane_id', 'label', 'coords']. The 'coords' column
                should contain lists of [x, y] coordinates.
        out_dir: The root directory to save the output 'arrow_boundaries' folder to.
        compression: The compression to use for the Feather files.
    """
    out_dir = Path(out_dir) / "viewer_data" / 'arrow_boundaries'
    out_dir.mkdir(parents=True, exist_ok=True)

    comp = compression if compression != "none" else None

    shards = []
    total_polys = 0
    total_points = 0

    # Define the schema once, to be used for all files
    schema = pa.schema([
        pa.field('x_list', pa.list_(pa.float32())),
        pa.field('y_list', pa.list_(pa.float32())),
        pa.field('plane_id', pa.uint16()),
        pa.field('label', pa.int32())
    ])

    # Process each DataFrame in the input list
    for idx, df_plane in enumerate(dfs_in):
        # Check for required columns
        required_cols = ['plane_id', 'cell_id', 'coords']
        if not all(col in df_plane.columns for col in required_cols):
            logger.info("Warning: A DataFrame is missing required columns. Skipping.")
            continue

        # Get the plane ID - use index as fallback for empty DataFrames
        if df_plane.empty:
            # Empty plane - use the list index as plane_id
            current_plane_id = idx
        else:
            # Non-empty plane - get plane_id from first row
            current_plane_id = int(df_plane['plane_id'].iloc[0])

            # Validate: plane_id from data should match list index
            if current_plane_id != idx:
                raise ValueError(
                    f"Plane ID mismatch: DataFrame at index {idx} has plane_id={current_plane_id}. "
                    f"Expected plane_id to match index. Check that dfs_in is ordered correctly by plane."
                )

        shard_name = f"boundaries_plane_{current_plane_id:02d}.feather"

        # Filter out rows with empty coordinate lists (skip if already empty)
        if not df_plane.empty:
            df_plane = df_plane.copy()
            df_plane = df_plane[df_plane["coords"].str.len() > 0]

        if df_plane.empty:
            # If plane has no valid polygons, write an empty Feather file
            empty_table = schema.empty_table()
            feather.write_feather(empty_table, (out_dir / shard_name).as_posix(), compression=comp)
            shards.append({"url": shard_name, "rows": 0, "plane": current_plane_id})
            # logger.info(f"Wrote empty shard {shard_name} for plane {current_plane_id}")
            continue

        # Prepare data for Arrow, using the 'coords' column directly
        x_lists = df_plane["coords"].apply(lambda coords: [float(x) for x, _ in coords])
        y_lists = df_plane["coords"].apply(lambda coords: [float(y) for _, y in coords])
        labels = pd.to_numeric(df_plane["cell_id"], errors="coerce").fillna(-1).astype("int32")

        # Create the Arrow table
        arrays = {
            "x_list": pa.array(x_lists.tolist(), type=pa.list_(pa.float32())),
            "y_list": pa.array(y_lists.tolist(), type=pa.list_(pa.float32())),
            "plane_id": pa.array([current_plane_id] * len(df_plane), type=pa.uint16()),
            "label": pa.array(labels.tolist(), type=pa.int32()),
        }
        table = pa.table(arrays, schema=schema)

        # Write the Feather file
        feather.write_feather(table, (out_dir / shard_name).as_posix(), compression=comp)

        polys = len(df_plane)
        pts = sum(x_lists.str.len())
        total_polys += polys
        total_points += pts

        shards.append({"url": shard_name, "rows": int(polys), "plane": current_plane_id})
        # print(f"Wrote {shard_name}: polys={polys}, points={pts}")

    # Manifest
    manifest = {
        "format": "arrow-feather",
        "total_rows": int(total_polys),
        "total_points": int(total_points),
        "shards": sorted(shards, key=lambda s: s['plane']),  # Sort shards by plane number
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # logger.info(f"Done. Total polys: {total_polys}. Total points: {total_points}. Files: {len(shards)}. Output: {out_dir}")
    logger.info(f"Saved at: {out_dir}")



def write_arrow(geneData:pd.DataFrame, cellData:pd.DataFrame, cellBoundaries:pd.DataFrame, out_dir: str = None) -> None:
    # cellBoundaries is one DataFrame per plane, so its length is how many planes
    # the spot shards have to cover, empty ones included
    geneData_to_arrow(geneData, out_dir, num_planes=len(cellBoundaries))
    cellData_to_arrow(cellData, out_dir)
    boundaries_to_arrow(cellBoundaries, out_dir)
