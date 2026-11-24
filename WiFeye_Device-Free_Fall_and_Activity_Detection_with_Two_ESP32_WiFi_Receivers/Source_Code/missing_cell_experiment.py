"""
WiFeye: Leave-one-cell-out test of subject-style fall augmentation
=======================================================================
Subject 3 recorded no falls. In any model trained on the full dataset, subject 3's
activities are present but their falls are not ("missing cell"), so the model may learn
the shortcut "looks like subject 3 -> not a fall". Subject 3 has no falls to test on, so
the effect of style augmentation can only be measured by SIMULATING the missing cell on
subjects that do have falls.

For each j in {1,2,4,5} (the simulated "subject 3") and each s != j (a fully unseen subject):
  train = all real trials EXCEPT j's falls and EXCEPT all of s's trials
          (j's non-fall activities stay in training, exactly like subject 3)
  test  = j's real falls (missing-cell sensitivity) + all of s's trials
          (false alarms on an unseen subject's ADL, and standard unseen-subject sensitivity)
Arms (same seeds, same model):
  NONE     no synthetic data
  GENERIC  generic augmented falls (sources in train)          -> "just more falls"
  STYLE    falls rescaled to j's motion-energy level (sources in train) -> the subject-3 idea
Synthetic data is only ever added to training. Pre-registered adoption rule (survey
report, P2-9): STYLE must raise missing-cell sensitivity by >= 5 points over NONE with a
bootstrap interval excluding 0, without raising unseen-subject false alarms.

Usage:
  python missing_cell_experiment.py --features-dir data/features_v2 \
      --generic-dir data/features_aug_v2 --style-dir-pattern data/features_aug_v2_s{j} \
      --out data/models_exp/missing_cell_v2 --seeds 3
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import train_classifier as tc

CLASSES = tc.CLASSES
ADL = ["walk", "sit", "stand", "hand"]


def pooled(df, X, mask, clf):
    p_raw = clf.predict_proba(X[mask])
    p = np.zeros((mask.sum(), len(CLASSES)))
    for k, c in enumerate(clf.classes_):
        p[:, CLASSES.index(c)] = p_raw[:, k]
    sub = df[mask].reset_index(drop=True)
    out = {}
    for tid, idx in sub.groupby("trial_id").indices.items():
        reg = idx[sub["in_region"].values[idx]]
        reg = reg if len(reg) else idx
        out[tid] = CLASSES[int(np.argmax(p[reg].mean(axis=0)))]
    return out


def main():
    ap = argparse.ArgumentParser(description="Leave-one-cell-out style-augmentation test")
    ap.add_argument("--features-dir", default="data/features_v2")
    ap.add_argument("--generic-dir", default="data/features_aug_v2")
    ap.add_argument("--style-dir-pattern", default="data/features_aug_v2_s{j}")
    ap.add_argument("--out", default="data/models_exp/missing_cell_v2")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-estimators", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, X, cols = tc.load_features(os.path.join(args.features_dir, "features_all_windows.csv"))
    df = tc.apply_event_labels(df, 1.0)
    gen_df, gen_X = tc.load_augmented(args.generic_dir, cols)
    gen_df = tc.apply_event_labels(gen_df, 1.0)
    gen_keep = (gen_df["kind"] == "generic").values
    gen_df, gen_X = gen_df[gen_keep].reset_index(drop=True), gen_X[gen_keep]
    trials = df.drop_duplicates("trial_id").set_index("trial_id")

    rows = []
    for j in ["1", "2", "4", "5"]:
        st_df, st_X = tc.load_augmented(args.style_dir_pattern.format(j=j), cols)
        st_df = tc.apply_event_labels(st_df, 1.0)
        for s in ["1", "2", "3", "4", "5"]:
            if s == j:
                continue
            j_falls = set(trials.index[(trials.subject == j) & (trials.activity == "fall")])
            s_all = set(trials.index[trials.subject == s])
            test_ids = j_falls | s_all
            tr = (~df["trial_id"].isin(test_ids)).values & df["is_action"].values
            te = df["trial_id"].isin(test_ids).values
            train_ids = set(df.loc[tr, "trial_id"])
            arms = {"NONE": (None, None), "GENERIC": (gen_df, gen_X), "STYLE": (st_df, st_X)}
            for seed in range(args.seeds):
                for arm, (adf, aX) in arms.items():
                    Xtr, ytr = X[tr], df["activity"].values[tr]
                    n_syn = 0
                    if adf is not None:
                        am = adf["source_trial"].isin(train_ids).values & adf["is_action"].values
                        n_syn = int(adf.loc[am, "trial_id"].nunique())
                        Xtr = np.vstack([Xtr, aX[am]]); ytr = np.concatenate([ytr, adf["activity"].values[am]])
                    clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=15,
                                                 class_weight="balanced", n_jobs=-1, random_state=seed)
                    clf.fit(Xtr, ytr)
                    pred = pooled(df, X, te, clf)
                    jf = [pred[t] == "fall" for t in j_falls]
                    s_adl = [t for t in s_all if trials.loc[t, "activity"] in ADL]
                    s_fall = [t for t in s_all if trials.loc[t, "activity"] == "fall"]
                    rows.append({"j": j, "s": s, "seed": seed, "arm": arm, "n_synthetic_trials": n_syn,
                                 "missing_cell_sens": float(np.mean(jf)),
                                 "unseen_far": float(np.mean([pred[t] == "fall" for t in s_adl])),
                                 "unseen_sens": float(np.mean([pred[t] == "fall" for t in s_fall])) if s_fall else np.nan})
            print(f"  j={j} s={s} done")

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(args.out, "missing_cell_runs.csv"), index=False)
    pair = R.groupby(["j", "s", "arm"])[["missing_cell_sens", "unseen_far", "unseen_sens"]].mean().reset_index()
    piv = pair.pivot_table(index=["j", "s"], columns="arm", values=["missing_cell_sens", "unseen_far"])
    rng = np.random.default_rng(0)
    out = {"by_arm": pair.groupby("arm")[["missing_cell_sens", "unseen_far", "unseen_sens"]].mean().round(4).to_dict(),
           "by_j": pair.groupby(["j", "arm"])["missing_cell_sens"].mean().unstack().round(3).to_dict()}
    print("\n=======================================================")
    print("  LEAVE-ONE-CELL-OUT: does style augmentation help a subject with no falls in training?")
    print("=======================================================")
    print(pair.groupby("arm")[["missing_cell_sens", "unseen_far", "unseen_sens"]].mean().round(3).to_string())
    print("\n  missing-cell fall sensitivity by simulated subject j:")
    print(pair.groupby(["j", "arm"])["missing_cell_sens"].mean().unstack().round(3).to_string())
    for arm in ("GENERIC", "STYLE"):
        for metric in ("missing_cell_sens", "unseen_far"):
            d = (piv[(metric, arm)] - piv[(metric, "NONE")]).values
            boots = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(5000)]
            ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            out[f"delta_{metric}_{arm}_vs_NONE"] = {"mean": float(d.mean()), "ci95": ci}
            print(f"  {arm:>7} - NONE  {metric:<18} mean {d.mean():+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    d_sens = (piv[("missing_cell_sens", "STYLE")] - piv[("missing_cell_sens", "NONE")]).values
    d_far = (piv[("unseen_far", "STYLE")] - piv[("unseen_far", "NONE")]).values
    ci_s = out["delta_missing_cell_sens_STYLE_vs_NONE"]["ci95"]
    adopt = (d_sens.mean() >= 0.05) and (ci_s[0] > 0) and (d_far.mean() <= 0.0 + 1e-9 or out["delta_unseen_far_STYLE_vs_NONE"]["ci95"][0] <= 0)
    out["adopt_style_augmentation"] = bool(adopt)
    print(f"\n  Pre-registered rule (>= +5 pts, CI > 0, no FAR increase): {'ADOPT' if adopt else 'DO NOT ADOPT'}")
    with open(os.path.join(args.out, "missing_cell_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {args.out}/")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
