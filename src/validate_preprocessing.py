"""Phase 2 validation: split integrity, test CSV contents and preprocessing behaviour.

Usage: python3 -m src.validate_preprocessing
Exits with a non-zero status if any check fails.
"""
import hashlib
import io
import sys

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, ID_COLUMN, NUMERICAL_FEATURES,
    RAW_DATA_PATH, TARGET, TEST_DATA_PATH, build_preprocessor, clean_features,
    load_raw_data, load_train_test_split, split_features_target,
)

# SHA-1 of the original CSV recorded during Phase 1 inspection.
ORIGINAL_SHA1 = "252c7d50740da7988d71174d15184247463d975c"

results = []


def check(name, passed, detail=""):
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    # 1. Original CSV unchanged.
    sha1 = hashlib.sha1(RAW_DATA_PATH.read_bytes()).hexdigest()
    check("Original CSV unchanged", sha1 == ORIGINAL_SHA1, f"sha1={sha1}")

    raw = load_raw_data()
    X_train, X_test, y_train, y_test = load_train_test_split()

    # 2. No overlap between train and test.
    shared_index = X_train.index.intersection(X_test.index)
    shared_ids = set(raw.loc[X_train.index, ID_COLUMN]) & set(raw.loc[X_test.index, ID_COLUMN])
    check("Train/test row indices do not overlap", len(shared_index) == 0,
          f"{len(shared_index)} shared")
    check("Train/test customerIDs do not overlap", len(shared_ids) == 0,
          f"{len(shared_ids)} shared")
    check("Train + test cover every row once", len(X_train) + len(X_test) == len(raw),
          f"{len(X_train)} + {len(X_test)} = {len(X_train) + len(X_test)} of {len(raw)}")

    # 3. Target proportions preserved.
    full_rate = (raw[TARGET] == "Yes").mean()
    for name, y in (("train", y_train), ("test", y_test)):
        check(f"Churn rate preserved in {name}", abs(y.mean() - full_rate) < 0.01,
              f"{name}={y.mean():.4f} vs full={full_rate:.4f}")

    # 4. Test CSV contents: 19 raw features + Churn, no customerID, raw values intact.
    test_csv = pd.read_csv(TEST_DATA_PATH, dtype=str, keep_default_na=False)
    check("Test CSV columns are the 19 features + Churn",
          list(test_csv.columns) == FEATURE_COLUMNS + [TARGET],
          f"{test_csv.shape[1]} columns")
    check("Test CSV excludes customerID", ID_COLUMN not in test_csv.columns)
    check("Test CSV row count matches split", len(test_csv) == len(X_test),
          f"{len(test_csv)} rows")
    raw_text = pd.read_csv(RAW_DATA_PATH, dtype=str, keep_default_na=False)
    expected = raw_text.loc[X_test.index, FEATURE_COLUMNS + [TARGET]].reset_index(drop=True)
    check("Test CSV values identical to raw CSV text", expected.equals(test_csv))
    X_csv, y_csv = split_features_target(pd.read_csv(TEST_DATA_PATH))
    check("Test CSV round-trips through split_features_target",
          y_csv.equals(y_test.reset_index(drop=True)))

    # 5. Missing TotalCharges handled.
    blanks_train = int((raw.loc[X_train.index, "TotalCharges"].str.strip() == "").sum())
    blanks_test = int((raw.loc[X_test.index, "TotalCharges"].str.strip() == "").sum())
    cleaned_train, cleaned_test = clean_features(X_train), clean_features(X_test)
    check("No missing TotalCharges after cleaning",
          cleaned_train["TotalCharges"].isna().sum() == 0
          and cleaned_test["TotalCharges"].isna().sum() == 0,
          f"blanks before: train={blanks_train}, test={blanks_test}")
    imputed = raw["TotalCharges"].str.strip() == ""
    check("Imputed TotalCharges rows all have tenure 0 and value 0",
          bool((raw.loc[imputed, "tenure"] == 0).all())
          and bool((clean_features(raw.loc[imputed])["TotalCharges"] == 0).all()))

    preprocessor = build_preprocessor().fit(X_train)
    Xt_train, Xt_test = preprocessor.transform(X_train), preprocessor.transform(X_test)
    check("No NaN in transformed train/test matrices",
          not np.isnan(Xt_train).any() and not np.isnan(Xt_test).any(),
          f"train {Xt_train.shape}, test {Xt_test.shape}")
    feature_names = preprocessor.get_feature_names_out()
    check("Transformed feature names available", len(feature_names) == Xt_train.shape[1],
          f"{len(feature_names)} encoded features")

    # 6. Unseen categories are ignored rather than raising.
    probe = X_test.iloc[[0]].copy()
    probe["PaymentMethod"] = "Cryptocurrency"
    probe["Contract"] = "Weekly"
    try:
        out = pd.DataFrame(preprocessor.transform(probe), columns=feature_names)
        groups_zero = all(out.filter(like=f"cat__{col}_").to_numpy().sum() == 0
                          for col in ("PaymentMethod", "Contract"))
        check("Unseen categories handled (encoded as all zeros)", groups_zero)
    except Exception as exc:  # noqa: BLE001 - report any failure as a check result
        check("Unseen categories handled (encoded as all zeros)", False, repr(exc))

    # 7. Preprocessing fitted on training data only.
    encoder = preprocessor.named_steps["encode"]
    scaler = encoder.named_transformers_["num"]
    ohe = encoder.named_transformers_["cat"]
    train_means = cleaned_train[NUMERICAL_FEATURES].mean().to_numpy()
    full_means = clean_features(raw)[NUMERICAL_FEATURES].mean().to_numpy()
    check("Scaler fitted on training rows only",
          scaler.n_samples_seen_ == len(X_train) and np.allclose(scaler.mean_, train_means)
          and not np.allclose(scaler.mean_, full_means),
          f"n_samples_seen_={scaler.n_samples_seen_}")
    train_categories = [sorted(cleaned_train[c].unique()) for c in CATEGORICAL_FEATURES]
    check("One-hot categories learned from training rows",
          [sorted(c) for c in ohe.categories_] == train_categories)

    # 8. Fitted pipeline can be saved/loaded for the app and reproduces its output.
    buffer = io.BytesIO()
    joblib.dump(preprocessor, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    check("Fitted preprocessor survives joblib round-trip",
          np.allclose(restored.transform(X_csv), Xt_test))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    print(f"Encoded features: {len(NUMERICAL_FEATURES)} numerical + "
          f"{len(feature_names) - len(NUMERICAL_FEATURES)} one-hot = {len(feature_names)}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
