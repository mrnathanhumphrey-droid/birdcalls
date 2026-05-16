"""Spatial-join xeno-canto recordings to BCR polygons + cell-availability report."""
import pathlib
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ROOT = pathlib.Path(r"D:/Bird Song")

print("[1] Loading xeno-canto metadata ...")
df = pd.read_parquet(ROOT / "range_data" / "xc_metadata_4species_conus.parquet")
df["lat_num"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon_num"] = pd.to_numeric(df["lon"], errors="coerce")
df["q_score"] = df["q"].fillna("no score")
df["type_norm"] = df["type"].astype(str).str.lower()
df_song = df[df["type_norm"].str.contains("song", na=False, regex=False)].copy()
df_song = df_song.dropna(subset=["lat_num","lon_num"])
df_song = df_song[~df_song["q_score"].isin(["E","e","no score"])].copy()
print(f"  usable: {len(df_song)}")

print("\n[2] Loading BCR polygons ...")
bcr_shp = ROOT / "range_data" / "bcr_shapefile" / "extracted" / "Data" / "SHPfiles" / "BCR_NA.shp"
bcr = gpd.read_file(bcr_shp)
print(f"  BCRs: {len(bcr)}  cols: {list(bcr.columns)}")
bcr = bcr.to_crs("EPSG:5070")
# Print BCR id col candidates
for c in bcr.columns:
    if c != "geometry":
        sample = bcr[c].head(3).tolist()
        print(f"    {c}: {sample}")

print("\n[3] Spatial join ...")
pts = gpd.GeoDataFrame(
    df_song[["common_name","scientific_name","ssp","lat_num","lon_num","id","q_score","length"]].copy(),
    geometry=[Point(lon, lat) for lon, lat in zip(df_song["lon_num"], df_song["lat_num"])],
    crs="EPSG:4326",
).to_crs("EPSG:5070")
joined = gpd.sjoin(pts, bcr, how="left", predicate="within")
n_match = joined["index_right"].notna().sum()
print(f"  matched: {n_match} / {len(pts)}")

# Detect BCR id + name columns
bcr_id_col = next((c for c in joined.columns if c.upper() in ["BCR","BCRNUMBER","BCRNUM","BCR_CODE","BCR_NUM","BCRNAME"] or "BCR" in c.upper() and "NUMBER" in c.upper()), None)
if bcr_id_col is None:
    bcr_id_col = next((c for c in joined.columns if "BCR" in c.upper()), None)
print(f"  using BCR col: {bcr_id_col}")

joined["bcr"] = joined[bcr_id_col]
joined_us = joined.dropna(subset=["bcr"]).copy()
print(f"  recordings with BCR: {len(joined_us)}")

print("\n[4] Per-(species, BCR) cell counts:")
cell_counts = joined_us.groupby(["common_name","bcr"]).size().unstack(fill_value=0).astype(int)
print(cell_counts.to_string())

print("\n[5] Cells passing thresholds:")
print(f"  N>=10 per species: {(cell_counts >= 10).sum(axis=1).to_dict()}")
print(f"  N>=20 per species: {(cell_counts >= 20).sum(axis=1).to_dict()}")
print(f"  N>=30 per species: {(cell_counts >= 30).sum(axis=1).to_dict()}")

# Total cells across all species at each threshold
for thresh in [10, 20, 30]:
    total_cells = (cell_counts >= thresh).sum().sum()
    print(f"  TOTAL (species, BCR) cells >= {thresh} recordings: {total_cells}")

# Save
cell_counts.to_csv(ROOT / "range_data" / "cell_counts_species_bcr.csv")
joined_us.drop(columns="geometry").to_parquet(ROOT / "range_data" / "xc_metadata_joined_bcr.parquet", index=False)
print(f"\n  saved cell_counts.csv + xc_metadata_joined_bcr.parquet")

# Detail BCRs covered
print("\n[6] BCRs covered (any species, any count):")
covered = cell_counts.sum(axis=0)
print(f"  total distinct BCRs touched: {(covered > 0).sum()} of {len(bcr)}")
top10 = covered.sort_values(ascending=False).head(10)
print(f"  top 10 BCRs by total recordings:")
for b, n in top10.items():
    print(f"    BCR {b}: {n}")
