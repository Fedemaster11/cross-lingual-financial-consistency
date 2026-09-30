
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
R1 = ROOT / "results" / "robustness" / "r1_max12_raw.csv"
R4 = ROOT / "results" / "robustness" / "r4_rationale_raw.csv"
OUT = ROOT / "results" / "robustness" / "r4_recovered"
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["en","es","de","fr","zh","ja","ar"]
VAL = {"A":-2,"B":-1,"C":0,"D":1,"E":2}
LEAD = re.compile(r"^\s*([A-Ea-e])(?:\b|\s|[|=:;,.+\-–—])")

def recover(row):
    if str(row.get("valid_format","")).lower() == "true":
        x = str(row.get("label","")).strip().upper()
        if x in VAL:
            return x
    m = LEAD.match(str(row.get("raw_output","")).strip())
    return m.group(1).upper() if m else np.nan

def prep(path):
    d = pd.read_csv(path)
    d["rec_label"] = d.apply(recover, axis=1)
    d["rec_value"] = d["rec_label"].map(VAL)
    return d

def deltas(d):
    p = d.pivot_table(
        index=["language","pair_id","family"],
        columns="condition",
        values=["rec_value","expected_label_value"],
        aggfunc="first"
    )
    p.columns = ["_".join(c) for c in p.columns]
    p = p.reset_index().dropna(subset=["rec_value_original","rec_value_counterfactual"])
    p["delta"] = p["rec_value_counterfactual"] - p["rec_value_original"]
    p["expected_delta"] = p["expected_label_value_counterfactual"] - p["expected_label_value_original"]
    p["direction_correct"] = np.sign(p["delta"]) == np.sign(p["expected_delta"])
    p["exact_delta"] = p["delta"] == p["expected_delta"]
    p["abs_delta"] = p["delta"].abs()
    return p

r1 = prep(R1)
r4 = prep(R4)

if len(r4) != 840:
    raise RuntimeError(f"Expected 840 R4 rows, found {len(r4)}")

strict = r4["valid_format"].astype(str).str.lower().eq("true").sum()
recoverable = r4["rec_label"].notna().sum()

m = r1[["task_id","language","rec_label"]].rename(columns={"rec_label":"r1"}).merge(
    r4[["task_id","rec_label"]].rename(columns={"rec_label":"r4"}),
    on="task_id"
).dropna()
m["agree"] = m["r1"] == m["r4"]

by_lang = m.groupby("language").agg(n=("agree","size"), agreement=("agree","mean")).reset_index()
by_lang["agreement_percent"] = 100 * by_lang["agreement"]
by_lang.to_csv(OUT/"r1_vs_r4_agreement_by_language.csv", index=False)

d4 = deltas(r4)
fam = d4.groupby("family").agg(
    n=("pair_id","size"),
    direction_correct=("direction_correct","mean"),
    exact_expected_delta=("exact_delta","mean"),
    mean_abs_delta=("abs_delta","mean")
).reset_index().sort_values("direction_correct", ascending=False)
fam.to_csv(OUT/"r4_family_summary.csv", index=False)

dist = r4.groupby(["language","rec_label"]).size().unstack(fill_value=0).reindex(
    index=LANGS, columns=["A","B","C","D","E"], fill_value=0
)
dist.to_csv(OUT/"r4_label_distribution.csv")

# R4 direct label agreement heatmap
mat = pd.DataFrame(np.nan, index=LANGS, columns=LANGS)
for a in LANGS:
    da = r4[r4.language==a][["pair_id","condition","rec_label"]].rename(columns={"rec_label":"a"})
    for b in LANGS:
        db = r4[r4.language==b][["pair_id","condition","rec_label"]].rename(columns={"rec_label":"b"})
        z = da.merge(db, on=["pair_id","condition"]).dropna()
        mat.loc[a,b] = (z["a"] == z["b"]).mean()

(mat*100).to_csv(OUT/"r4_direct_label_agreement_percent.csv")

fig, ax = plt.subplots(figsize=(8,7))
im = ax.imshow((mat*100).values)
ax.set_xticks(range(7), [x.upper() for x in LANGS])
ax.set_yticks(range(7), [x.upper() for x in LANGS])
ax.set_title("R4 — Direct Cross-Lingual Label Agreement")
ax.set_xlabel("Language")
ax.set_ylabel("Language")
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{mat.iloc[i,j]*100:.1f}%", ha="center", va="center")
fig.colorbar(im, ax=ax, label="Same recovered A–E label (%)")
fig.tight_layout()
fig.savefig(OUT/"r4_direct_label_agreement.png", dpi=220, bbox_inches="tight")
plt.close(fig)

summary = pd.DataFrame([{
    "rows": len(r4),
    "strict_format_valid": strict,
    "strict_format_rate": strict/len(r4),
    "recoverable_labels": recoverable,
    "recoverable_rate": recoverable/len(r4),
    "complete_deltas": len(d4),
    "direction_correct_rate": d4["direction_correct"].mean(),
    "exact_expected_delta_rate": d4["exact_delta"].mean(),
    "mean_abs_delta": d4["abs_delta"].mean(),
    "r1_r4_label_agreement": m["agree"].mean(),
}])
summary.to_csv(OUT/"r4_summary.csv", index=False)

print("="*72)
print("R4 RECOVERED-LABEL ANALYSIS")
print("="*72)
print(f"Strict format:             {strict}/840 ({strict/840:.1%})")
print(f"Recoverable leading label: {recoverable}/840 ({recoverable/840:.1%})")
print(f"Complete deltas:           {len(d4)}/420")
print(f"Direction correct:         {d4['direction_correct'].mean():.1%}")
print(f"Exact expected delta:      {d4['exact_delta'].mean():.1%}")
print(f"Mean |delta|:              {d4['abs_delta'].mean():.3f}")
print(f"R1 vs R4 label agreement:  {m['agree'].mean():.1%}")
print()
print("R1 vs R4 agreement by language:")
print(by_lang.to_string(index=False, formatters={"agreement_percent":lambda x:f"{x:.1f}%"}))
print()
print("R4 family summary:")
print(fam.to_string(index=False))
print()
print("Saved to:", OUT)
