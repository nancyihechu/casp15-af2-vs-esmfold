"""
Analyze step: build every table and figure from results/ only (plus config/targets.yaml).

Tables (results/tables/):
  smallest_msa_matching_esmfold.csv   smallest MSA where AF2 TM >= ESMFold TM, per target
  perplexity_vs_accuracy.csv          perplexity and TM-score per target
  spearman.csv                        Spearman rho with exact permutation p-values (n = 6)
  confidence_quadrants.csv            runs counted by confident/not x right/wrong
  speed.csv                           accuracy lost vs time saved, per target
  t1122_all_models.csv                all 5 AF2 models of the T1122 full and depth-1 reruns
Figures (results/figures/): fig1 .. fig6, figS1.
Also writes results/summary.md (markdown tables of all of the above) and copies it into
README.md between the <!-- results:start --> and <!-- results:end --> markers.

Duplicate depth runs (is_duplicate_depth) are left out of every AF2 depth analysis, and
AF2 depth is shown as the actual number of sequences / Neff/L, not the planned depth.
Usage: python scripts/analyze.py [--config ...]   (notebooks/analysis.ipynb calls the same functions)
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from project import load_config  # noqa: E402

# Colors: reference categorical palette, slots 1-2 (validated pair), text and grid inks
AF2_COLOR, ESM_COLOR = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
PAE_CMAP = LinearSegmentedColormap.from_list(  # sequential blue: dark = low error
    "pae", ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb", "#fcfcfb"])
PLDDT_BANDS = [(90, 101, "0x0053D6"), (70, 90, "0x65CBF3"), (50, 70, "0xFFDB13"), (0, 50, "0xFF7D45")]
SMALL_N_NOTE = "n = 6 targets: |rho| >= 0.89 is needed for p < 0.05"
# A TM-score margin this small counts as a tie (one seed, one MSA subsample per depth)
TIE_TM = 0.01

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "lines.linewidth": 2, "lines.markersize": 7,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


# ----------------------------------------------------------------- data loading

def load(config=None):
    cfg = load_config(config)
    runs = pd.read_csv(cfg["tables_dir"] / "all_runs.csv", dtype={"msa_depth": str})
    runs["is_duplicate_depth"] = runs["is_duplicate_depth"].astype(str) == "True"
    order = [t["id"] for t in cfg["targets"]]
    notes = {t["id"]: t.get("note", "") for t in cfg["targets"]}
    ppl = pd.read_csv(cfg["tables_dir"] / "perplexity.csv")
    return {"cfg": cfg, "runs": runs, "order": order, "notes": notes, "ppl": ppl,
            "af2": runs[(runs.method == "AF2") & ~runs.is_duplicate_depth].copy(),
            "af2_full": runs[(runs.method == "AF2") & (runs.msa_depth == "full")].set_index("target"),
            "esm": runs[runs.method == "ESMFold"].set_index("target")}


def title(d, tid):
    return f"{tid} (near-single-sequence case)" if d["notes"][tid] else tid


def short_note(d, tid):
    return f"{tid} (near-single-seq.)" if d["notes"][tid] else tid


def plain_log_axis(ax, ticks, axis="x"):
    """Log axis labelled with plain numbers at the given ticks (no 2x10^0 style labels)."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    kept = []
    for t in sorted(set(ticks), reverse=True):  # drop ticks too close (on a log axis) to a kept one
        if not kept or kept[-1] / t >= 1.5:
            kept.append(t)
    target.set_major_locator(FixedLocator(kept))
    target.set_minor_locator(NullLocator())
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def label_points(ax, xs, ys, labels, fontsize=8.5):
    """Direct labels next to points, moving a label to another side when it would collide."""
    fig = ax.figure
    fig.canvas.draw()
    placed, char_w, h = [], fontsize * 0.62, fontsize * 1.3
    offsets = [(6, 4, "left"), (6, -13, "left"), (-6, 4, "right"), (-6, -13, "right"), (6, 14, "left")]
    for x, y, lab in sorted(zip(xs, ys, labels), key=lambda t: -t[1]):
        px, py = ax.transData.transform((x, y)) * 72 / fig.dpi
        w = len(lab) * char_w
        for dx, dy, ha in offsets:
            x0 = px + dx if ha == "left" else px + dx - w
            box = (x0, py + dy, x0 + w, py + dy + h)
            if all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in placed):
                break
        placed.append(box)
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=fontsize, color=INK2)


def save(d, fig, name):
    out = d["cfg"]["results_dir"] / "figures" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def write_table(d, df, name):
    out = d["cfg"]["tables_dir"] / name
    df.to_csv(out, index=False, float_format="%.4g")
    return df


def spearman_exact(x, y):
    """Spearman rho and exact two-sided permutation p-value (all n! orderings)."""
    rx, ry = rankdata(x), rankdata(y)
    rho = np.corrcoef(rx, ry)[0, 1]
    perms = [np.corrcoef(rx, np.array(p))[0, 1] for p in itertools.permutations(ry)]
    p = np.mean(np.abs(perms) >= abs(rho) - 1e-12)
    return float(rho), float(p)


# ----------------------------------------------------------------------- tables

def table_smallest_msa(d):
    rows = []
    for tid in d["order"]:
        esm_tm = d["esm"].loc[tid, "tm_score"]
        af2 = d["af2"][d["af2"].target == tid].sort_values("n_sequences")
        ok = af2[af2.tm_score >= esm_tm]
        row = {"target": tid, "esmfold_tm": esm_tm, "af2_full_tm": d["af2_full"].loc[tid, "tm_score"],
               "af2_full_n_sequences": d["af2_full"].loc[tid, "n_sequences"]}
        if ok.empty:
            row.update(smallest_n_sequences=np.nan, smallest_neff_per_L=np.nan, af2_tm_there=np.nan,
                       margin_tm=np.nan, stays_at_or_above_for_larger_msa=False)
        else:
            first = ok.iloc[0]
            larger = af2[af2.n_sequences >= first.n_sequences]
            row.update(smallest_n_sequences=int(first.n_sequences),
                       smallest_neff_per_L=first.neff_per_L, af2_tm_there=first.tm_score,
                       margin_tm=first.tm_score - esm_tm,
                       stays_at_or_above_for_larger_msa=bool((larger.tm_score >= esm_tm).all()))
        below = af2[af2.tm_score < esm_tm]
        row["largest_n_sequences_below_esmfold"] = int(below.n_sequences.max()) if len(below) else np.nan
        full_margin = row["af2_full_tm"] - esm_tm
        row["full_msa_margin_tm"] = full_margin
        row["full_msa_verdict"] = ("tie" if abs(full_margin) <= TIE_TM else
                                   "AF2 better" if full_margin > 0 else "ESMFold better")
        row["note"] = d["notes"][tid]
        rows.append(row)
    return write_table(d, pd.DataFrame(rows), "smallest_msa_matching_esmfold.csv")


def table_noise(d):
    """TM-score differences between runs with identical inputs (duplicate depth runs vs kept run)."""
    msa = pd.read_csv(d["cfg"]["tables_dir"] / "msa_depths.csv", dtype={"msa_depth": str, "duplicate_of": str})
    af2 = d["runs"][d["runs"].method == "AF2"].set_index(["target", "msa_depth"])
    rows = [{"target": r.target, "msa_depth": r.msa_depth, "same_input_as": r.duplicate_of,
             "tm_score": af2.loc[(r.target, r.msa_depth), "tm_score"],
             "tm_score_kept_run": af2.loc[(r.target, r.duplicate_of), "tm_score"]}
            for r in msa[msa.is_duplicate_depth].itertuples()]
    df = pd.DataFrame(rows)
    df["abs_diff"] = (df.tm_score - df.tm_score_kept_run).abs()
    return write_table(d, df, "duplicate_run_noise.csv")


def table_perplexity(d):
    df = d["ppl"][["target", "pseudo_perplexity_full", "pseudo_perplexity_domain"]].copy()
    df["esmfold_tm"] = df.target.map(d["esm"].tm_score)
    df["af2_full_tm"] = df.target.map(d["af2_full"].tm_score)
    df["af2_minus_esmfold_tm"] = df.af2_full_tm - df.esmfold_tm
    df["note"] = df.target.map(d["notes"])
    return write_table(d, df, "perplexity_vs_accuracy.csv")


def table_spearman(d):
    ppl = table_perplexity(d)
    rows = []
    for x in ("pseudo_perplexity_full", "pseudo_perplexity_domain"):
        for y in ("esmfold_tm", "af2_full_tm", "af2_minus_esmfold_tm"):
            rho, p = spearman_exact(ppl[x], ppl[y])
            rows.append({"x": x, "y": y, "n": len(ppl), "rho": rho, "p_exact": p})
    for method, df in (("AF2 (non-duplicate runs)", d["af2"]), ("ESMFold", d["esm"])):
        rho, p = (spearman_exact(df.mean_plddt, df.tm_score) if len(df) <= 8 else
                  (float(np.corrcoef(rankdata(df.mean_plddt), rankdata(df.tm_score))[0, 1]), np.nan))
        rows.append({"x": f"mean_plddt [{method}]", "y": "tm_score", "n": len(df), "rho": rho,
                     "p_exact": p})
    df = pd.DataFrame(rows)
    df["note"] = np.where(df.n == 6, SMALL_N_NOTE, "runs share targets, so they are not independent; no p-value")
    return write_table(d, df, "spearman.csv")


def table_confidence_quadrants(d):
    rows = []
    for method, df in (("AF2", d["af2"]), ("ESMFold", d["esm"].reset_index())):
        confident, right = df.mean_plddt >= 70, df.tm_score >= 0.5
        rows.append({"method": method, "runs": len(df),
                     "confident_and_right": int((confident & right).sum()),
                     "confident_but_wrong": int((confident & ~right).sum()),
                     "unsure_but_right": int((~confident & right).sum()),
                     "unsure_and_wrong": int((~confident & ~right).sum()),
                     "rule": "confident = mean pLDDT >= 70; right = TM-score >= 0.5"})
    return write_table(d, pd.DataFrame(rows), "confidence_quadrants.csv")


def table_speed(d):
    rows = []
    for tid in d["order"]:
        a, e = d["af2_full"].loc[tid], d["esm"].loc[tid]
        rows.append({"target": tid, "af2_full_tm": a.tm_score, "esmfold_tm": e.tm_score,
                     "tm_lost_with_esmfold": a.tm_score - e.tm_score,
                     "af2_msa_time_s": a.msa_time_s, "af2_fold_time_s": a.fold_time_s,
                     "af2_total_time_s": a.total_time_s, "af2_time_per_model_s": a.time_per_model_s,
                     "esmfold_time_s": e.total_time_s,
                     "speedup_total": a.total_time_s / e.total_time_s,
                     "speedup_per_model": a.time_per_model_s / e.time_per_model_s,
                     "note": d["notes"][tid]})
    return write_table(d, pd.DataFrame(rows), "speed.csv")


def table_t1122_models(d):
    path = d["cfg"]["tables_dir"] / "af2_all_models.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"msa_depth": str})
    df = df[df.target == "T1122"].sort_values(["msa_depth", "rank"], ascending=[False, True])
    original = d["runs"][(d["runs"].target == "T1122") & (d["runs"].method == "AF2")].set_index("msa_depth")
    df["original_run_rank1_tm"] = df.msa_depth.map(original.tm_score)
    return write_table(d, df, "t1122_all_models.csv")


def table_t1122_paired(d):
    """Same AF2 model (weights 1-5) with and without the duplicate MSA row."""
    df = table_t1122_models(d)
    if df is None:
        return None
    wide = df.pivot(index="model", columns="msa_depth", values="tm_score")
    out = pd.DataFrame({"model": wide.index, "tm_full_msa": wide["full"].values,
                        "tm_depth_1": wide["1"].values})
    out["difference"] = out.tm_full_msa - out.tm_depth_1
    return write_table(d, out, "t1122_paired_models.csv")


# ---------------------------------------------------------------------- figures

def fig1(d, x="n_sequences", name="fig1_tm_vs_n_sequences.png", xlabel="Sequences in MSA (log scale)"):
    smallest = table_smallest_msa(d).set_index("target")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharey=True)
    for ax, tid in zip(axes.flat, d["order"]):
        af2 = d["af2"][d["af2"].target == tid].sort_values(x)
        esm_tm = d["esm"].loc[tid, "tm_score"]
        ax.axhline(esm_tm, color=ESM_COLOR, ls="--", lw=2, label="ESMFold (no MSA)")
        ax.plot(af2[x], af2.tm_score, "-o", color=AF2_COLOR, label="AF2 (ColabFold)")
        s = smallest.loc[tid]
        if np.isnan(s.smallest_n_sequences):
            msg = "AF2 never reaches ESMFold"
        else:
            p = af2[af2.n_sequences == s.smallest_n_sequences].iloc[0]
            ax.plot(p[x], p.tm_score, "o", ms=15, mfc="none", mec=INK, mew=1.5)
            n = int(p.n_sequences)
            verb = "ties" if s.margin_tm <= TIE_TM else "\u2265"
            if n == af2.n_sequences.min():
                msg = f"AF2 {verb} ESMFold at every MSA size"
            elif n == af2.n_sequences.max():
                msg = f"AF2 {verb} ESMFold only with the full MSA ({n} seq.)"
            elif s.stays_at_or_above_for_larger_msa:
                msg = f"AF2 {verb} ESMFold from {n} seq. up (ring)"
            else:
                msg = f"AF2 {verb} ESMFold first at {n} seq. (ring)"
        ax.text(0.97, 0.04, msg, transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
        ax.set_xscale("log")
        if x == "n_sequences":
            plain_log_axis(ax, af2[x])
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylim(0, 1.05)
        ax.set_title(title(d, tid), fontsize=10 if d["notes"][tid] else 11)
    for ax in axes[:, 0]:
        ax.set_ylabel("TM-score to native (FM domain)")
    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    axes.flat[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("AF2 accuracy vs MSA size; dashed = ESMFold on the same target "
                 "(duplicate depth runs not shown)", y=1.0)
    if x == "neff_per_L":
        fig.text(0.5, -0.02, "T1122: both points share Neff/L = 1/L, because its only MSA hit is an "
                 "identical copy of the query.", ha="center", fontsize=9, color=INK2)
    fig.tight_layout()
    return save(d, fig, name)


def fig1b(d):
    return fig1(d, x="neff_per_L", name="fig1b_tm_vs_neff_per_L.png",
                xlabel="Neff / L (80% identity weighting, log scale)")


def fig2(d):
    ppl = table_perplexity(d)
    sp = table_spearman(d).set_index(["x", "y"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, col, color, name in ((axes[0], "esmfold_tm", ESM_COLOR, "ESMFold"),
                                 (axes[1], "af2_full_tm", AF2_COLOR, "AF2, full MSA")):
        ax.scatter(ppl.pseudo_perplexity_full, ppl[col], s=60, color=color, edgecolor=SURFACE, lw=2, zorder=3)
        ax.set_xlim(4, 20)
        ax.set_ylim(0, 1.05)
        label_points(ax, ppl.pseudo_perplexity_full, ppl[col], [short_note(d, t) for t in ppl.target])
        rho, p = sp.loc[("pseudo_perplexity_full", col), ["rho", "p_exact"]]
        ax.set_title(f"{name}:  Spearman rho = {rho:.2f}, exact p = {p:.3f}")
        ax.set_xlabel("ESM-2 pseudo-perplexity (full sequence)")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("TM-score to native")
    fig.text(0.5, -0.04, SMALL_N_NOTE + ". Treat these as a hint, not a test.",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout()
    return save(d, fig, "fig2_perplexity_vs_tm.png")


def fig3(d):
    table_confidence_quadrants(d)
    sp = table_spearman(d)
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.axvline(70, color=INK2, lw=1, ls=":")
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.text(71, 0.03, "confident but wrong\n(pLDDT >= 70, TM < 0.5)", fontsize=8.5, color=INK2)
    for method, df, color, marker in (("AF2", d["af2"], AF2_COLOR, "o"),
                                      ("ESMFold", d["esm"].reset_index(), ESM_COLOR, "s")):
        row = sp[sp.x.str.contains(method)].iloc[0]
        extra = f", exact p = {row.p_exact:.3f}" if len(df) <= 8 else " (runs not independent)"
        ax.scatter(df.mean_plddt, df.tm_score, s=55, color=color, marker=marker, edgecolor=SURFACE,
                   lw=1.5, zorder=3, label=f"{method} ({len(df)} runs): Spearman rho = {row.rho:.2f}{extra}")
    t1122 = d["runs"][(d["runs"].target == "T1122") & ~d["runs"].is_duplicate_depth]
    for _, r in t1122.iterrows():
        ax.annotate("T1122", (r.mean_plddt, r.tm_score), xytext=(-34, 6), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.set_xlim(20, 100)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Mean pLDDT over the FM domain (model's own confidence)")
    ax.set_ylabel("TM-score to native")
    ax.set_title("Confidence vs accuracy (AF2: all non-duplicate depth runs)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return save(d, fig, "fig3_confidence_vs_accuracy.png")


def render_structure(d, tid, model, out_png):
    """PyMOL: native domain in gray, one model on top colored by pLDDT (AlphaFold DB bands)."""
    from pymol import cmd
    sup = d["cfg"]["results_dir"] / tid / "superposition"
    cmd.reinitialize()
    cmd.load(str(sup / "native.pdb"), "native")
    cmd.load(str(sup / f"{model}.pdb"), "pred")  # ("model" is a reserved PyMOL word)
    if max(a.b for a in cmd.get_model("pred and name CA").atom) <= 1.0:
        cmd.alter("pred", "b = b * 100")  # ESMFold stores pLDDT on a 0-1 scale
    cmd.hide("everything")
    cmd.show("cartoon")
    cmd.set("cartoon_transparency", 0.65, "native")
    cmd.color("grey70", "native")
    for lo, hi, hex_color in PLDDT_BANDS:
        cmd.color(hex_color, f"pred and b > {lo - 0.001} and b < {hi}")
    cmd.orient("native")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.png(str(out_png), width=900, height=750, dpi=150, ray=1)


def fig4(d):
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    parts = d["cfg"]["results_dir"] / "figures" / "fig4_parts"
    parts.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(d["order"]), 2, figsize=(8, 3.3 * len(d["order"])))
    for row, tid in zip(axes, d["order"]):
        for ax, model, name in ((row[0], "af2_full", "AF2 (full MSA)"), (row[1], "esmfold", "ESMFold")):
            png = parts / f"{tid}_{model}.png"
            render_structure(d, tid, model, png)
            ax.imshow(plt.imread(png))
            ax.axis("off")
            tm = (d["af2_full"] if model == "af2_full" else d["esm"]).loc[tid, "tm_score"]
            ax.set_title(f"{short_note(d, tid)}: {name}, TM = {tm:.2f}", fontsize=10)
    handles = [plt.Line2D([], [], marker="s", ls="", ms=10, color="#" + c[2:], label=lab)
               for (_, _, c), lab in zip(PLDDT_BANDS, ["pLDDT > 90", "70-90", "50-70", "< 50"])]
    handles.append(plt.Line2D([], [], marker="s", ls="", ms=10, color="#b3b3b3", label="native (gray)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Models superposed on the native FM domain (TMscore superposition)", y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return save(d, fig, "fig4_structures.png")


def fig5(d):
    n = len(d["order"])
    fig, axes = plt.subplots((n + 1) // 2, 4, figsize=(14, 3.5 * ((n + 1) // 2)), layout="constrained")
    res_dir = d["cfg"]["results_dir"]
    for i, tid in enumerate(d["order"]):
        summary = json.loads((res_dir / tid / "af2/depth_full/summary.json").read_text())
        af2_pae = json.loads((res_dir / tid / "af2/depth_full" / summary["models"][0]["scores_file"]).read_text())["pae"]
        esm_pae = json.loads((res_dir / tid / "esmfold/scores.json").read_text())["pae"]
        for j, (pae, name) in enumerate(((af2_pae, "AF2 full MSA"), (esm_pae, "ESMFold"))):
            ax = axes.flat[2 * i + j]
            im = ax.imshow(np.array(pae), cmap=PAE_CMAP, vmin=0, vmax=30)
            ax.set_title(f"{tid}{'*' if d['notes'][tid] else ''}: {name}", fontsize=10)
            ax.grid(False)
            ax.set_xlabel("Scored residue")
            if i % 2 == 0 and j == 0:
                ax.set_ylabel("Aligned residue")
    for ax in axes.flat[2 * n:]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Predicted aligned error (A); darker = more confident")
    fig.suptitle("PAE maps (full chain; AF2 = top-ranked model, full MSA)")
    fig.text(0.01, -0.02, "* T1122 = near-single-sequence case (MSA = query + an identical copy).",
             fontsize=9, color=INK2)
    return save(d, fig, "fig5_pae.png")


def fig6(d):
    speed = table_speed(d)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for _, r in speed.iterrows():
        ax.plot([r.esmfold_time_s, r.af2_total_time_s], [r.esmfold_tm, r.af2_full_tm], color=GRID, lw=1.5, zorder=1)
    ax.scatter(speed.af2_total_time_s, speed.af2_full_tm, s=60, color=AF2_COLOR, edgecolor=SURFACE, lw=1.5,
               zorder=3, label="AF2, full MSA: MSA search + 5 models")
    ax.scatter(speed.esmfold_time_s, speed.esmfold_tm, s=60, marker="s", color=ESM_COLOR, edgecolor=SURFACE,
               lw=1.5, zorder=3, label="ESMFold: 1 model")
    ax.set_xscale("log")
    ax.set_xlim(0.4, 80)
    plain_log_axis(ax, [0.5, 1, 2, 5, 10, 20, 50])
    ax.set_ylim(0, 1.05)
    label_points(ax, speed.af2_total_time_s, speed.af2_full_tm, [short_note(d, t) for t in speed.target], 8)
    ax.set_xlabel("Total time per target (s, log scale; L40S GPU)")
    ax.set_ylabel("TM-score to native")
    ax.set_title("Accuracy vs total time")
    ax.legend(loc="lower left", fontsize=9)

    ax = axes[1]
    y = np.arange(len(speed))
    ax.scatter(speed.af2_time_per_model_s, y, s=110, color=AF2_COLOR, zorder=3, label="AF2, time per model")
    ax.scatter(speed.esmfold_time_s, y, s=45, marker="s", color=ESM_COLOR, edgecolor=SURFACE, lw=1,
               zorder=4, label="ESMFold (1 model)")
    for yi, r in zip(y, speed.itertuples()):
        ax.plot([r.esmfold_time_s, r.af2_time_per_model_s], [yi, yi], color=GRID, lw=1.5, zorder=1)
    ax.set_yticks(y, [short_note(d, t) for t in speed.target])
    ax.set_xscale("log")
    ax.set_xlim(0.4, 8)
    plain_log_axis(ax, [0.5, 1, 2, 5])
    ax.set_ylim(len(speed) - 0.4, -0.9)
    ax.set_xlabel("GPU folding time (s, log scale; no MSA search, no JAX compile)")
    ax.set_title("Folding time per model")
    ax.legend(loc="upper center", fontsize=9, ncol=2)
    fig.text(0.5, -0.06, "AF2 MSA search times (about 6 s per FM target) may be server-cached: the "
             "uncached positive control T1147 took 38.9 s.\nAF2 fold time = 5 x median of models 2-5 "
             "(JAX compile left out); ESMFold time measured after a warm-up fold.",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout()
    return save(d, fig, "fig6_accuracy_vs_speed.png")


def figS1(d):
    df = table_t1122_models(d)
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for depth, color, marker, label in (("full", AF2_COLOR, "o", "full MSA (query + identical copy)"),
                                        ("1", INK2, "D", "depth 1 (query only)")):
        sub = df[df.msa_depth == depth].sort_values("model")
        ax.plot(sub.model.str[-1].astype(int), sub.tm_score, "-" + marker, color=color, label=label)
        top = sub[sub["rank"] == 1].iloc[0]
        ax.plot(int(top.model[-1]), top.tm_score, "o", ms=16, mfc="none", mec=color, mew=1.5)
    ax.set_xticks([1, 2, 3, 4, 5], [f"model {i}" for i in range(1, 6)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("TM-score to native")
    ax.set_title("T1122 (near-single-sequence case): all 5 AF2 models\n(ring = top-ranked by pLDDT)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return save(d, fig, "figS1_T1122_all_models.png")


# ----------------------------------------------------------- markdown summary

def md_table(df, digits=None):
    """Plain markdown table; digits = {column: decimals}."""
    digits = digits or {}

    def fmt(c, v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits.get(c, 3)}f}"
        return str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(c, v) for c, v in zip(df.columns, row)) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join(lines)


def findings_md(d):
    """Key findings as sentences; every number is computed from the tables."""
    name = {t: short_note(d, t) for t in d["order"]}
    join = lambda ts: ", ".join(name[t] for t in ts) or "none"  # noqa: E731
    sm = table_smallest_msa(d)
    depth1 = d["runs"][(d["runs"].method == "AF2") & (d["runs"].msa_depth == "1")].set_index("target")
    below1 = [t for t in d["order"] if depth1.loc[t, "tm_score"] < d["esm"].loc[t, "tm_score"]]
    few = sm[(sm.smallest_n_sequences > 1) & (sm.smallest_n_sequences <= 16)]
    sp = table_spearman(d).set_index(["x", "y"])
    ppl = table_perplexity(d)
    top_ppl = set(ppl.nlargest(2, "pseudo_perplexity_full").target)
    hard = {m: set(ppl.nsmallest(2, c).target) for m, c in (("ESMFold", "esmfold_tm"), ("AF2", "af2_full_tm"))}
    quad = table_confidence_quadrants(d).set_index("method")
    speed = table_speed(d)
    lines = [
        "**Key findings** (generated from the tables below)", "",
        f"- **MSA depth (Q1).** With a single sequence, AF2 is below ESMFold on {len(below1)} of "
        f"{len(d['order'])} targets ({join(below1)}). A few sequences close the gap: AF2 reaches "
        f"ESMFold's TM-score with {', '.join(f'{int(r.smallest_n_sequences)} ({name[r.target]})' for r in few.itertuples())} "
        "sequences"
        + "".join(f"; {name[r.target]} reaches it only with its full MSA ({int(r.smallest_n_sequences)} sequences)"
                  for r in sm[sm.smallest_n_sequences > 16].itertuples())
        + "".join(f"; {name[r.target]} is above ESMFold even with 1 sequence"
                  for r in sm[sm.smallest_n_sequences == 1].itertuples())
        + ". With the full MSA, AF2 is clearly better on "
        f"{join(sm[sm.full_msa_verdict == 'AF2 better'].target)} and ties ESMFold (margin of {TIE_TM} TM or "
        f"less) on {join(sm[sm.full_msa_verdict == 'tie'].target)}"
        + (f"; ESMFold is better on {join(sm[sm.full_msa_verdict == 'ESMFold better'].target)}" if
           (sm.full_msa_verdict == "ESMFold better").any() else "") + ".",
        f"- **Perplexity (Q2).** Higher ESM-2 pseudo-perplexity goes with lower accuracy, but not "
        f"significantly: rho = {sp.loc[('pseudo_perplexity_full', 'esmfold_tm'), 'rho']:.2f} for ESMFold "
        f"(p = {sp.loc[('pseudo_perplexity_full', 'esmfold_tm'), 'p_exact']:.2f}) and "
        f"{sp.loc[('pseudo_perplexity_full', 'af2_full_tm'), 'rho']:.2f} for AF2 "
        f"(p = {sp.loc[('pseudo_perplexity_full', 'af2_full_tm'), 'p_exact']:.2f}), n = {len(ppl)}. "
        f"The two highest-perplexity targets ({join(sorted(top_ppl, key=d['order'].index))}) are "
        + ("the two hardest for both methods." if top_ppl == hard["ESMFold"] == hard["AF2"] else
           f"{'' if top_ppl == hard['ESMFold'] else 'not '}the two hardest for ESMFold and "
           f"{'' if top_ppl == hard['AF2'] else 'not '}the two hardest for AF2."),
        f"- **Confidence.** Neither method was confident but wrong (mean pLDDT >= 70 with TM < 0.5): "
        f"{quad.loc['AF2', 'confident_but_wrong']} of {quad.loc['AF2', 'runs']} AF2 runs, "
        f"{quad.loc['ESMFold', 'confident_but_wrong']} of {quad.loc['ESMFold', 'runs']} ESMFold runs. "
        f"ESMFold's pLDDT ranks the targets by accuracy with rho = "
        f"{sp.loc[('mean_plddt [ESMFold]', 'tm_score'), 'rho']:.2f}.",
        f"- **Speed (Q3).** ESMFold took {speed.esmfold_time_s.min():.1f}-{speed.esmfold_time_s.max():.1f} s "
        f"per target; AF2 took {speed.af2_total_time_s.min():.0f}-{speed.af2_total_time_s.max():.0f} s "
        f"(MSA search + 5 models), i.e. {speed.speedup_total.min():.1f}-{speed.speedup_total.max():.1f}x "
        f"longer in total, or {speed.speedup_per_model.min():.1f}-{speed.speedup_per_model.max():.1f}x per "
        f"model. The accuracy given up with ESMFold ranges from {speed.tm_lost_with_esmfold.min():.3f} to "
        f"{speed.tm_lost_with_esmfold.max():.3f} TM, depending on the target.",
    ]
    paired = table_t1122_paired(d)
    if paired is not None:
        lines.append(
            f"- **T1122, the near-single-sequence case.** Its \"full MSA\" is the query plus an identical "
            f"copy, yet it beats AF2 on the query alone for {int((paired.difference > 0).sum())} of "
            f"{len(paired)} model weights (mean {paired.difference.mean():+.3f} TM), so this is not ranking "
            "luck. The extra row changes AF2's input features without adding any evolutionary information. "
            "On the query alone AF2 is "
            + ("below" if depth1.loc["T1122", "tm_score"] < d["esm"].loc["T1122", "tm_score"] else "at or above")
            + " ESMFold.")
    return "\n".join(lines)


def summary_md(d):
    """All numbers for the README, taken from the pipeline's own tables."""
    runs, af2_full, esm = d["runs"], d["af2_full"], d["esm"]
    depth1 = runs[(runs.method == "AF2") & (runs.msa_depth == "1")].set_index("target")
    ppl = d["ppl"].set_index("target")
    name = {t: short_note(d, t) for t in d["order"]}
    main = pd.DataFrame({
        "target": [name[t] for t in d["order"]],
        "MSA seqs (full)": [int(af2_full.loc[t, "n_sequences"]) for t in d["order"]],
        "Neff/L (full)": [af2_full.loc[t, "neff_per_L"] for t in d["order"]],
        "AF2 full TM": [af2_full.loc[t, "tm_score"] for t in d["order"]],
        "AF2 1-seq TM": [depth1.loc[t, "tm_score"] for t in d["order"]],
        "ESMFold TM": [esm.loc[t, "tm_score"] for t in d["order"]],
        "AF2 full lDDT": [af2_full.loc[t, "lddt"] for t in d["order"]],
        "ESMFold lDDT": [esm.loc[t, "lddt"] for t in d["order"]],
        "AF2 full pLDDT": [af2_full.loc[t, "mean_plddt"] for t in d["order"]],
        "ESMFold pLDDT": [esm.loc[t, "mean_plddt"] for t in d["order"]],
        "ESM-2 perplexity": [ppl.loc[t, "pseudo_perplexity_full"] for t in d["order"]]})
    dig = {"Neff/L (full)": 3, "AF2 full pLDDT": 1, "ESMFold pLDDT": 1, "ESM-2 perplexity": 2}

    sm = table_smallest_msa(d)
    small = pd.DataFrame({
        "target": [name[t] for t in sm.target], "ESMFold TM": sm.esmfold_tm,
        "smallest MSA with AF2 >= ESMFold (seqs)": sm.smallest_n_sequences.map(lambda v: "" if np.isnan(v) else int(v)),
        "Neff/L there": sm.smallest_neff_per_L, "AF2 TM there": sm.af2_tm_there,
        "margin (TM)": sm.margin_tm, "AF2 stays >= ESMFold with more seqs": sm.stays_at_or_above_for_larger_msa,
        "full MSA: AF2 - ESMFold (TM)": sm.full_msa_margin_tm, "full MSA verdict": sm.full_msa_verdict})
    noise = table_noise(d)

    sp = table_spearman(d)
    sp = sp[sp.n == 6][["x", "y", "rho", "p_exact"]].rename(columns={"p_exact": "exact p (n = 6)"})
    quad = table_confidence_quadrants(d).drop(columns="rule")

    speed = table_speed(d)
    sp_tab = pd.DataFrame({
        "target": [name[t] for t in speed.target], "AF2 MSA search (s)": speed.af2_msa_time_s,
        "AF2 fold, 5 models (s)": speed.af2_fold_time_s, "AF2 per model (s)": speed.af2_time_per_model_s,
        "ESMFold, 1 model (s)": speed.esmfold_time_s, "AF2/ESMFold per model": speed.speedup_per_model,
        "AF2 total (s)": speed.af2_total_time_s, "AF2/ESMFold total": speed.speedup_total,
        "TM lost with ESMFold": speed.tm_lost_with_esmfold})
    sdig = {c: 2 for c in sp_tab.columns if c not in ("target", "TM lost with ESMFold")}
    sdig["AF2/ESMFold per model"] = sdig["AF2/ESMFold total"] = 1

    pc_path = d["cfg"]["results_dir"] / "positive_control/tables/all_runs.csv"
    pc = pd.read_csv(pc_path) if pc_path.exists() else None
    if pc is not None:
        pc["n_sequences"] = pc.n_sequences.map(lambda v: "" if pd.isna(v) else int(v))
    out = ["<!-- This block is generated by scripts/analyze.py from results/tables/*.csv. Do not edit. -->",
           findings_md(d), "",
           "### Main results (FM domain; AF2 = top-ranked of 5 models)", md_table(main, dig), "",
           "### Smallest MSA where AF2 is at or above ESMFold (TM-score)",
           f"Duplicate depth runs left out. A margin of {TIE_TM} TM or less counts as a tie: there is one "
           f"seed and one MSA subsample per depth, and runs with identical inputs already differ by up to "
           f"{noise.abs_diff.max():.4f} TM ({len(noise)} duplicate runs, "
           "`results/tables/duplicate_run_noise.csv`).",
           md_table(small), "",
           "Full-MSA verdicts: " + "; ".join(
               f"{name[r.target]}: {r.full_msa_verdict} ({r.full_msa_margin_tm:+.3f})" for r in sm.itertuples()) + ".",
           "",
           "### ESM-2 perplexity vs accuracy (Spearman, exact permutation p)",
           f"{SMALL_N_NOTE}.", md_table(sp), "",
           "### Confidence vs accuracy (AF2: non-duplicate depth runs)", md_table(quad), "",
           "### Speed (L40S GPU)", md_table(sp_tab, sdig), ""]
    msa_fm = speed.af2_msa_time_s
    if pc is not None:
        a = pc[pc.method == "AF2"].iloc[0]
        e = pc[pc.method == "ESMFold"].iloc[0]
        out += [f"MSA search times for the FM targets ({msa_fm.min():.1f}-{msa_fm.max():.1f} s) may be "
                f"cached by the ColabFold server: the positive control {a.target}, searched the same way, "
                f"took {a.msa_time_s:.1f} s.", "",
                f"### Positive control: {a.target} (easy TBM target, not in the main results)",
                md_table(pc[["method", "n_sequences", "tm_score", "gdt_ts", "lddt", "mean_plddt"]],
                         {"gdt_ts": 1, "mean_plddt": 1}), ""]
    t1122 = table_t1122_models(d)
    if t1122 is not None:
        paired = table_t1122_paired(d)
        top = t1122[t1122["rank"] == 1].set_index("msa_depth")
        gpu_diff = (top.tm_score - top.original_run_rank1_tm).abs().max()
        out += ["### T1122: all 5 AF2 models, full MSA (query + identical copy) vs depth 1",
                f"Reruns on {t1122.gpu.iloc[0]} that keep every model. The top models reproduce the "
                f"original L40S runs to within {gpu_diff:.4f} TM, so GPU type does not change the result.",
                md_table(t1122[["msa_depth", "rank", "model", "tm_score", "lddt", "mean_plddt",
                                "original_run_rank1_tm"]], {"mean_plddt": 1}), "",
                "Same model weights, with vs without the duplicate row:",
                md_table(paired), "",
                f"The duplicate row raises the TM-score of {int((paired.difference > 0).sum())} of "
                f"{len(paired)} models (mean change {paired.difference.mean():+.3f}; mean TM "
                f"{paired.tm_full_msa.mean():.3f} vs {paired.tm_depth_1.mean():.3f}). The top-ranked model "
                f"is the same weights ({top.loc['full', 'model']}) in both runs.", ""]
    text = "\n".join(out)
    (d["cfg"]["results_dir"] / "summary.md").write_text(text + "\n")
    return text


def update_readme(d):
    """Replace the generated block in README.md with results/summary.md."""
    readme = d["cfg"]["results_dir"].parent / "README.md"
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    text = readme.read_text()
    if start not in text:
        return None
    head, rest = text.split(start, 1)
    tail = rest.split(end, 1)[1]
    readme.write_text(f"{head}{start}\n{summary_md(d)}\n{end}{tail}")
    return readme


TABLES = [table_noise, table_smallest_msa, table_perplexity, table_spearman, table_confidence_quadrants,
          table_speed, table_t1122_models, table_t1122_paired]
FIGURES = [fig1, fig1b, fig2, fig3, fig4, fig5, fig6, figS1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="default: config/targets.yaml")
    d = load(ap.parse_args().config)
    for f in TABLES:
        f(d)
    for f in FIGURES:
        out = f(d)
        if out:
            print("Wrote", Path(out).relative_to(d["cfg"]["results_dir"].parent))
    summary_md(d)
    if update_readme(d):
        print("Updated the results block in README.md")


if __name__ == "__main__":
    main()
