"""Read-only inspection of the Telco Customer Churn CSV.

Usage: python3 src/inspection/inspect_dataset.py [path/to/csv]
The CSV is only read; nothing is written back to disk.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"
ID_COL = "customerID"


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    # Read once with pandas' default inference, and once as raw strings
    # so blank/whitespace cells can be detected before any type coercion.
    df = pd.read_csv(csv_path)
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    section("1. File")
    print(f"Path: {csv_path.resolve()}")

    section("2. Shape")
    print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")

    section("3. Columns and inferred dtypes")
    for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
        print(f"{i:>2}. {col:<18} {dtype}")

    section("4. Numerical vs categorical (as inferred by pandas)")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    print(f"Numerical ({len(num_cols)}): {num_cols}")
    print(f"Non-numeric ({len(cat_cols)}): {cat_cols}")
    binary_numeric = [c for c in num_cols if df[c].nunique() <= 2]
    print(f"Numeric columns that are actually binary flags: {binary_numeric}")

    section("5. Missing / null / blank values per column")
    nulls = df.isna().sum()
    blanks = raw.apply(lambda s: s.str.strip().eq("").sum())
    missing = pd.DataFrame({"pandas_null": nulls, "blank_or_whitespace": blanks})
    missing["total"] = missing.sum(axis=1)
    print(missing.to_string())
    print(f"\nTotal pandas nulls: {int(nulls.sum())}   "
          f"Total blank/whitespace cells: {int(blanks.sum())}")

    section("6. Duplicates")
    print(f"Fully duplicated rows: {int(df.duplicated().sum())}")
    print(f"Duplicate {ID_COL} values: {int(df[ID_COL].duplicated().sum())}")
    dup_wo_id = df.drop(columns=[ID_COL]).duplicated().sum()
    print(f"Rows duplicated when ignoring {ID_COL}: {int(dup_wo_id)}")

    section("7. Unique-value counts")
    for col in df.columns:
        n = df[col].nunique(dropna=False)
        if n <= 10:
            vals = df[col].value_counts(dropna=False).to_dict()
            print(f"{col:<18} {n:>5}  {vals}")
        else:
            print(f"{col:<18} {n:>5}")

    section(f"8. Target distribution: {TARGET}")
    counts = df[TARGET].value_counts()
    pct = df[TARGET].value_counts(normalize=True) * 100
    print(pd.DataFrame({"count": counts, "percent": pct.round(2)}).to_string())

    section("9. TotalCharges check")
    tc_raw = raw["TotalCharges"]
    tc_num = pd.to_numeric(tc_raw, errors="coerce")
    bad = tc_num.isna()
    print(f"Inferred dtype: {df['TotalCharges'].dtype}")
    print(f"Values that fail numeric conversion: {int(bad.sum())}")
    print(f"Distinct failing raw values: {sorted(set(repr(v) for v in tc_raw[bad]))}")
    print("\nRows with non-numeric TotalCharges:")
    print(df.loc[bad, [ID_COL, "tenure", "MonthlyCharges", "TotalCharges", TARGET]]
          .to_string())
    print(f"\nRows with tenure == 0: {int((df['tenure'] == 0).sum())}")
    print(f"tenure == 0 rows coincide with blank TotalCharges: "
          f"{bool((df['tenure'] == 0).eq(bad).all())}")

    # Sanity check: TotalCharges should be roughly tenure * MonthlyCharges.
    ok = ~bad
    ratio = tc_num[ok] / (df.loc[ok, "tenure"] * df.loc[ok, "MonthlyCharges"])
    print(f"TotalCharges / (tenure * MonthlyCharges): "
          f"median={ratio.median():.3f}, min={ratio.min():.3f}, max={ratio.max():.3f}")
    corr = df.loc[ok, ["tenure", "MonthlyCharges"]].assign(TotalCharges=tc_num[ok]).corr()
    print("\nCorrelation among numeric features (TotalCharges converted):")
    print(corr.round(3).to_string())

    section("10. Numeric summary")
    summary = df[["tenure", "MonthlyCharges"]].assign(TotalCharges=tc_num).describe()
    print(summary.round(2).to_string())

    section("11. Other data-quality checks")
    str_cols = raw.columns
    ws = {c: int((raw[c] != raw[c].str.strip()).sum()) for c in str_cols}
    print(f"Cells with leading/trailing whitespace: "
          f"{ {c: n for c, n in ws.items() if n} or 'none'}")
    no_internet = [c for c in df.columns
                   if df[c].astype(str).eq("No internet service").any()]
    no_phone = [c for c in df.columns
                if df[c].astype(str).eq("No phone service").any()]
    print(f"Columns containing 'No internet service': {no_internet}")
    print(f"Columns containing 'No phone service': {no_phone}")
    internet_no = df["InternetService"].eq("No")
    consistent = all(df.loc[internet_no, c].eq("No internet service").all()
                     and not df.loc[~internet_no, c].eq("No internet service").any()
                     for c in no_internet)
    print(f"'No internet service' exactly matches InternetService == 'No': {consistent}")

    section("12. Assignment requirement check")
    features = [c for c in df.columns if c not in (TARGET, ID_COL)]
    print(f"Instances: {len(df)}  (need >= 500) -> {'PASS' if len(df) >= 500 else 'FAIL'}")
    print(f"Input features excl. {ID_COL} and {TARGET}: {len(features)}  "
          f"(need >= 12) -> {'PASS' if len(features) >= 12 else 'FAIL'}")
    print(f"Features: {features}")


if __name__ == "__main__":
    main()
