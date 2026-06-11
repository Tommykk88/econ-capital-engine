"""Build the catastrophe event catalog used by the model.

Preferred source: EM-DAT (emdat.be, free academic registration). Export
storms + floods worldwide as xlsx and place the file at
`data/raw/emdat.xlsx`; this script will clean it (filter perils, drop events
without damage estimates, inflate to current dollars with a flat CPI factor
per year) and write `data/processed/cat_catalog.csv`.

Fallback: if no EM-DAT export is present, a SYNTHETIC pseudo-catalog is
generated, calibrated loosely to the order of magnitude of published insured
storm/flood losses (Swiss Re sigma): ~12 material events/year with a
lognormal body and a Pareto-type tail. The output file is flagged with a
`source` column so downstream docs can state clearly which one was used.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "cat_catalog.csv"

N_YEARS = 45
SEED = 7


def from_emdat(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[df["Disaster Type"].isin(["Storm", "Flood"])]
    df = df.dropna(subset=["Total Damage, Adjusted ('000 US$)"])
    df = df.rename(columns={"Start Year": "year"})
    df["loss"] = df["Total Damage, Adjusted ('000 US$)"] / 1_000.0  # -> USD millions
    df = df[df["loss"] > 100.0]  # ignore immaterial events (< $100m)
    df["source"] = "emdat"
    return df[["year", "loss", "source"]]


def synthetic() -> pd.DataFrame:
    """Pseudo-catalog: Poisson(12) events/yr, lognormal body + Pareto tail."""
    rng = np.random.default_rng(SEED)
    rows = []
    final_year = 2025
    for year in range(final_year - N_YEARS + 1, final_year + 1):
        for _ in range(rng.poisson(12)):
            if rng.uniform() < 0.12:  # tail event: Pareto, xi ~ 0.5
                loss = 2_000.0 * (rng.pareto(2.0) + 1.0)
            else:  # attritional event
                loss = rng.lognormal(mean=6.0, sigma=1.0)
            rows.append({"year": year, "loss": loss, "source": "synthetic"})
    return pd.DataFrame(rows)


def main() -> None:
    emdat_path = RAW / "emdat.xlsx"
    if emdat_path.exists():
        catalog = from_emdat(emdat_path)
        print(f"Built catalog from EM-DAT export: {len(catalog)} events")
    else:
        catalog = synthetic()
        print(
            f"No {emdat_path} found -> generated SYNTHETIC catalog "
            f"({len(catalog)} events over {N_YEARS} years). "
            "Register at emdat.be for the real data."
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
