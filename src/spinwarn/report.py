"""
The results table, the plots and the README's results section, from metrics.json.

    python -m spinwarn.report

Reads results/metrics.json, which the final run wrote, and nothing else, so
it runs on a fresh clone without data/ or results/cv/. It writes:

- results/results_table.md: every method on the test sessions at each
  threshold set on CV, both recalls with their bootstrap intervals; the
  headline per kind (natural, deliberate); each model against B3 and B2 on
  the same spins; and each method's config, PR-AUC and time per prediction;
- results/recall_vs_fa.png: every method's recall against false alarms per
  minute on CV, the headline beside the Project Plan's rule. The test
  sessions have no curve: metrics.json keeps them only at the two
  thresholds;
- results/importance.png: the best model's permutation importance on the
  test sessions;
- the README's section between RESULTS_START and RESULTS_END: the same
  tables and both plots. Everything else in the README is left as it is.

The paired comparison is the only number not in metrics.json already: each
model's headline catches against a baseline's on the same test spins, with
a bootstrap interval that resamples the spins once for both.
"""

import argparse
import json
import math

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from spinwarn.baselines import BASELINES
from spinwarn.config import (
    IMPORTANCE_PLOT_PATH, METRICS_PATH, README_PATH, RECALL_FA_PLOT_PATH, RESULTS_TABLE_PATH, SEED,
)
from spinwarn.cv import METHODS
from spinwarn.evaluate import warning_range
from spinwarn.models import MODELS


RESULTS_START, RESULTS_END = "<!-- results:start -->", "<!-- results:end -->"
# What each method is called in the tables and plots; B2's θ1 comes from its params.
NAMES = {"b1": "B1: |slip|", "b2": "B2: slip rate, |slip| > {theta1}°", "b3": "B3: time to 90°",
         "lr": "Logistic regression", "rf": "Random forest", "hgb": "Gradient boosting"}
# Each model is compared spin by spin with the two baselines that beat B1 on
# CV: B3, the one the Project Plan says to beat, then B2.
PAIRED_BASELINES = ("b3", "b2")
MAX_FA_PER_MIN = 3.0      # the recall plot's x range: both targets and what's around them
DPI = 150

# The dataviz skill's reference palette, light mode: its first six
# categorical slots in their fixed order (validated: adjacent CVD ΔE ≥ 9.1),
# one per method in METHODS order, so a method keeps its color in every plot.
# Three sit under 3:1 on the surface, so the tables carry every number too.
COLORS = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
SURFACE, INK, INK_2, MUTED, AXIS, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e1e0d9"
STYLE = {
    "font.size": 9, "text.color": INK, "axes.titlecolor": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
}
DASH = "—"                # a number metrics.json has as null: undefined, like a missed spin's lead


# --- numbers as text ----------------------------------------------------------

def method_name(method, params) -> str:
    """What the tables and plots call method: B2 with its θ1."""
    return NAMES[method].format(**params)


def _number(x, spec) -> str:
    """x formatted, with a true minus sign, or DASH for null."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return DASH
    text = format(x, spec)
    # What rounds to 0 prints as 0, so −0.00003 to 3 places is 0.000, not −0.000.
    if float(text.rstrip("%")) == 0:
        text = format(0.0, spec)
    return text.replace("-", "−")


def _interval(ci) -> str:
    """' [0.81, 1.00]', or nothing when the interval is null (no spins to resample)."""
    lo, hi = ci
    return "" if lo is None or hi is None else f" [{_number(lo, '.2f')}, {_number(hi, '.2f')}]"


def recall_cell(point, headline=True) -> str:
    """'0.93 (25/27) [0.81, 1.00]': a recall, the count behind it and its bootstrap interval.

    headline is recall with at least min_lead_s of lead; otherwise the
    Project Plan's rule, any catch. point is an operating point, or one of
    its kinds.
    """
    key, caught = ("recall_min_lead", "caught_min_lead") if headline else ("recall", "caught")
    return f"{_number(point[key], '.2f')} ({point[caught]}/{point['n_spins']}){_interval(point[f'{key}_ci'])}"


def ms_cell(timing) -> str:
    """The median ms of one prediction, to 3 figures, and on one core too where final timed it so."""
    text = _number(timing["ms"], ".3g")
    if "ms_n_jobs_1" in timing:
        text += f" ({_number(timing['ms_n_jobs_1'], '.3g')} on one core)"
    return text


def _row(cells) -> str:
    """A markdown table row; a | in a cell is escaped, as experiments.md's rows are."""
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


def _table(header, rows) -> list:
    return [_row(header), "| " + " | ".join("---" for _ in header) + " |", *(_row(r) for r in rows)]


# --- the paired comparison ----------------------------------------------------

def paired_difference(a, b, n=1000, seed=SEED) -> tuple:
    """(difference, lo, hi): recall of a minus recall of b on the same spins, and a paired 95% interval.

    a and b are one bool per spin, in the same order. Each of the n
    resamples draws the spins with replacement once, for both, as
    bootstrap_recall draws them for one method, so a spin both catch or
    both miss moves neither recall against the other.
    """
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape or not len(a):
        raise ValueError(f"need one catch per spin for both, on at least one spin: got {len(a)} and {len(b)}")
    picks = np.random.default_rng(seed).choice(len(a), size=(n, len(a)))
    lo, hi = np.percentile(a[picks].mean(axis=1) - b[picks].mean(axis=1), [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def paired(model_point, baseline_point) -> dict:
    """A model's headline catches against a baseline's, both at one target, on the same test spins.

    only_model and only_baseline count the spins one catches with at least
    min_lead_s of lead and the other doesn't; difference and ci are
    paired_difference's. ValueError if the two scored different spins.
    """
    a, b = model_point["spins"], baseline_point["spins"]
    if [(s["session"], s["t"]) for s in a] != [(s["session"], s["t"]) for s in b]:
        raise ValueError("the two methods' operating points list different spins")
    x = np.array([s["caught_min_lead"] for s in a], dtype=bool)
    y = np.array([s["caught_min_lead"] for s in b], dtype=bool)
    difference, lo, hi = paired_difference(x, y)
    return {"only_model": int((x & ~y).sum()), "only_baseline": int((~x & y).sum()),
            "difference": difference, "ci": [lo, hi]}


# --- the tables ---------------------------------------------------------------

def _points(result) -> list:
    return result["test"]["operating_points"]


def _targets(metrics) -> list:
    """The false-alarm targets, in the order every method's operating points come in."""
    return [p["fa_target"] for p in _points(next(iter(metrics["methods"].values())))]


def intro(metrics) -> str:
    """The paragraph before the tables: which run, which sessions, how the thresholds and intervals were set."""
    test, dev = metrics["test"], metrics["dev"]
    once = ("which scored the test sessions once" if not metrics["forced"] else
            "forced: the test sessions had been scored before, as experiments.md logs")
    sessions = ", ".join(f"`{s}` ({v['kind']}, {v['spins']} spins)" for s, v in test["sessions"].items())
    per_spin = f"; one test spin is {100 / test['spins']:.1f} points of recall" if test["spins"] else ""
    return (f"From `results/metrics.json`: the final run of {metrics['started'][:10]} at the "
            f"{metrics['horizon']:g} s horizon, {once}. The test sessions hold {test['spins']} spins in "
            f"{test['scored_min']:.1f} scored min: {sessions}. Each method's thresholds were set on CV, out of "
            f"fold over {len(dev['sessions'])} dev sessions ({dev['spins']} spins in {dev['scored_min']:.1f} "
            "scored min), and each model was refit on all of them. Recall counts a spin caught with at least "
            f"{metrics['min_lead_s']:g} s of lead (the headline) or with any lead from 0.1 s (the Project Plan's "
            "rule). Intervals are bootstrap 95%, the test spins resampled 1,000 times"
            f"{per_spin}.")


def operating_point_table(metrics, k) -> list:
    """Every method at its k-th threshold from CV, scored on the test sessions."""
    lead = f"{metrics['min_lead_s']:g}"
    header = ["Method", f"Recall, lead ≥ {lead} s", "Recall, any lead", "FA/min on test", "Alert precision",
              "Time in alarm", "Median lead (s)"]
    rows = []
    for method, result in metrics["methods"].items():
        p = _points(result)[k]
        rows.append([method_name(method, result["params"]), recall_cell(p), recall_cell(p, headline=False),
                     _number(p["fa_per_min"], ".2f"), _number(p["alert_precision"], ".2f"),
                     _number(p["time_in_alarm"], ".1%"), _number(p["median_lead"], ".2f")])
    return _table(header, rows)


def kinds_table(metrics) -> list:
    """The headline per kind of spin at every threshold, with the false-alarm rate in that kind's sessions."""
    lead = f"{metrics['min_lead_s']:g}"
    header = ["Method", "Spins"]
    for target in _targets(metrics):
        header += [f"@ {target:g} FA/min: recall, lead ≥ {lead} s", "FA/min"]
    rows = []
    for method, result in metrics["methods"].items():
        name = method_name(method, result["params"])
        for j, kind in enumerate(_points(result)[0]["by_kind"]):
            first = _points(result)[0]["by_kind"][kind]
            few = "" if first["enough"] else f", under {metrics['min_kind_spins']}"
            row = [name if j == 0 else "", f"{kind} ({first['n_spins']}{few})"]
            for p in _points(result):
                row += [recall_cell(p["by_kind"][kind]), _number(p["by_kind"][kind]["fa_per_min"], ".2f")]
            rows.append(row)
    return _table(header, rows)


def paired_table(metrics) -> list:
    """Each model against B3 and against B2, spin by spin, at every threshold."""
    header = ["Model", "Against"]
    for target in _targets(metrics):
        header += [f"@ {target:g} FA/min: model only / baseline only", "Difference in recall"]
    methods = metrics["methods"]
    rows = []
    for baseline in PAIRED_BASELINES:
        for model in (m for m in methods if m in MODELS):
            row = [method_name(model, methods[model]["params"]), method_name(baseline, methods[baseline]["params"])]
            for mp, bp in zip(_points(methods[model]), _points(methods[baseline])):
                c = paired(mp, bp)
                lo, hi = c["ci"]
                row += [f"{c['only_model']} / {c['only_baseline']}",
                        f"{_number(c['difference'], '+.2f')} [{_number(lo, '.2f')}, {_number(hi, '.2f')}]"]
            rows.append(row)
    return _table(header, rows)


def methods_table(metrics) -> list:
    """Each method's chosen config, how it was chosen, its PR-AUC on CV and on test, and its time per prediction."""
    header = ["Method", "Params", "Chosen by", "PR-AUC, CV → test", "ms per prediction"]
    rows = []
    for method, result in metrics["methods"].items():
        params = f"`{json.dumps(result['params'], ensure_ascii=False)}`" if result["params"] else DASH
        rows.append([method_name(method, result["params"]), params, result["chosen_by"],
                     f"{_number(result['cv']['pr_auc'], '.3f')} → {_number(result['test']['pr_auc'], '.3f')}",
                     ms_cell(result["timing"])])
    return _table(header, rows)


def tables(metrics, level) -> list:
    """Every table with its heading (at markdown heading level) and the sentence it needs, as lines."""
    h = "#" * level
    n_spins, lead = metrics["test"]["spins"], f"{metrics['min_lead_s']:g}"
    lines = [intro(metrics)]
    for k, target in enumerate(_targets(metrics)):
        lines += ["", f"{h} At the {target:g} FA/min threshold", "",
                  f"Each method at its threshold for {target:g} false alarm{'s' * (target != 1)} a minute: the "
                  "lowest that kept to it on CV. FA/min on test is the rate it gave on the test sessions.", "",
                  *operating_point_table(metrics, k)]
    lines += ["", f"{h} Natural and deliberate spins", "",
              f"The headline per kind. FA/min is the rate in that kind's test sessions. A kind with under "
              f"{metrics['min_kind_spins']} spins says so.", "", *kinds_table(metrics)]
    lines += ["", f"{h} Each model against B3 and B2, spin by spin", "",
              f"On the same {n_spins} test spins, by the headline: the spins one catches with at least {lead} s "
              "of lead and the other doesn't, and the difference in recall with a paired bootstrap 95% interval "
              "(both methods on the same 1,000 resamples of the spins). Each method runs at its own threshold "
              "from CV, so at its own false-alarm rate on test, as in the tables above.", "",
              *paired_table(metrics)]
    lines += ["", f"{h} Configs, PR-AUC and time per prediction", "",
              "PR-AUC is average precision on the scored frames: out of fold on CV, and the refit model on test. "
              "Time per prediction is the median of single-row predictions on scored test frames, timed by "
              "the final run.", "", *methods_table(metrics)]
    return lines


def results_table(metrics) -> str:
    """results/results_table.md."""
    return "\n".join(["# Results", "", "Written by `python -m spinwarn.report` from `metrics.json`. The README's "
                      "results section has the same tables, and the plots.", "", *tables(metrics, level=2)]) + "\n"


def _relative(path) -> str:
    """path as the README links it."""
    return path.relative_to(README_PATH.parent).as_posix()


def readme_block(metrics) -> str:
    """What goes between the README's markers: the tables, then both plots with what they show."""
    ranked = metrics["importance"]
    name = method_name(ranked["method"], metrics["methods"][ranked["method"]]["params"])
    lines = [f"<!-- Written by `python -m spinwarn.report` from results/metrics.json: edits between the "
             "markers are overwritten. -->", "", *tables(metrics, level=3)]
    lines += ["", "### Recall against false alarms, on CV", "",
              f"![Recall against false alarms per minute on CV, every method]({_relative(RECALL_FA_PLOT_PATH)})", "",
              f"Out of fold over the {len(metrics['dev']['sessions'])} dev sessions ({metrics['dev']['spins']} "
              "spins). At each false-alarm rate, the recall of the threshold `threshold_for_fa` sets for that "
              "rate, as it set the final run's for each target (the vertical lines): the lowest within the rate, "
              "from the threshold with the most false alarms up. Below that one, alarms run together into fewer "
              "and longer alerts, down to the alarm left on, which catches every spin; no threshold is set "
              "there. The test sessions have no curve: `metrics.json` keeps them only at the two thresholds."]
    lines += ["", "### Permutation importance", "",
              f"![Permutation importance of {name.lower()} on the test sessions]({_relative(IMPORTANCE_PLOT_PATH)})",
              "", f"{name}, the models' best CV PR-AUC, on the scored test frames: how far average precision "
              f"falls when one feature is shuffled, mean ± SD over {ranked['n_repeats']} shuffles. Features "
              "that carry the same information share it, so a short bar doesn't mean the information is unused."]
    return "\n".join(lines)


def replace_block(text, block) -> str:
    """text with everything between RESULTS_START and RESULTS_END replaced by block.

    ValueError unless each marker is there exactly once, start first, so a
    README without them is refused rather than written over.
    """
    if text.count(RESULTS_START) != 1 or text.count(RESULTS_END) != 1:
        raise ValueError(f"the README needs {RESULTS_START} and {RESULTS_END} once each, around the results")
    start, end = text.index(RESULTS_START) + len(RESULTS_START), text.index(RESULTS_END)
    if end < start:
        raise ValueError(f"{RESULTS_END} comes before {RESULTS_START} in the README")
    return f"{text[:start]}\n{block}\n{text[end:]}"


# --- the plots ----------------------------------------------------------------

def curve_steps(curve, key) -> tuple:
    """(fa_per_min, recall): a CV curve as the recall at the threshold threshold_for_fa sets for each rate.

    It sets the lowest threshold within the rate in evaluate.warning_range,
    from the threshold with the most false alarms up, so a candidate there is
    on the curve only if every lower one there gives more false alarms; they
    don't always fall as the threshold rises, since an alert can split in
    two. The rates come out rising, and between two the recall is the lower
    one's, as matplotlib's steps-post draws it. key is "recall_min_lead" or
    "recall".
    """
    warnings = warning_range(curve)
    fa = warnings["fa_per_min"].to_numpy(dtype=float)
    recall = warnings[key].to_numpy(dtype=float)
    below = np.minimum.accumulate(np.r_[np.inf, fa[:-1]])   # the fewest false alarms of any lower threshold
    on = fa < below
    return fa[on][::-1], recall[on][::-1]


def recall_at(curve, key, rate) -> float:
    """The curve's recall at a false-alarm rate: what threshold_for_fa's pick for that rate gives."""
    fa, recall = curve_steps(curve, key)
    return float(recall[np.searchsorted(fa, rate, side="right") - 1])


def recall_figure(metrics) -> Figure:
    """Every method's CV recall against false alarms per minute: the headline left, the Project Plan's rule right."""
    lead = f"{metrics['min_lead_s']:g}"
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(10, 4.4), layout="constrained")
        axes = fig.subplots(1, 2, sharey=True)
        panels = [("recall_min_lead", f"Lead ≥ {lead} s, the headline"), ("recall", "Any lead, the Project Plan's rule")]
        for ax, (key, title) in zip(axes, panels):
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for target in _targets(metrics):
                ax.axvline(target, color=AXIS, linewidth=0.8, zorder=1)
                ax.annotate(f"{target:g} FA/min", (target, 1.0), xytext=(3, 6), textcoords="offset points",
                            xycoords=("data", "axes fraction"), color=MUTED, fontsize=8)
            for method, result in metrics["methods"].items():
                fa, recall = curve_steps(result["cv"]["curve"], key)
                # Dashed baselines, solid models: a second cue beside the color.
                ax.plot(fa, recall, drawstyle="steps-post", color=COLORS[method], linewidth=1.6,
                        linestyle="--" if method in BASELINES else "-", label=method_name(method, result["params"]),
                        zorder=2)
            ax.set_xlim(0, MAX_FA_PER_MIN)
            ax.set_ylim(0, 1.02)
            ax.set_title(title, loc="left", fontsize=10, pad=18)
            ax.set_xlabel("False alarms per minute")
        axes[0].set_ylabel("Recall")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=8.5)
        fig.suptitle(f"Recall against false alarms, CV over {len(metrics['dev']['sessions'])} dev sessions "
                     f"({metrics['dev']['spins']} spins)", x=0.01, ha="left", fontsize=11)
    return fig


def importance_figure(metrics) -> Figure:
    """The best model's permutation importance, one bar per feature, highest first, each with its value."""
    ranked = metrics["importance"]
    features = ranked["features"]
    means = np.array([f["mean"] for f in features])
    stds = np.array([f["std"] for f in features])
    y = np.arange(len(features))
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(7.5, 1.4 + 0.24 * len(features)), layout="constrained")
        ax = fig.subplots()
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.barh(y, means, height=0.6, color=COLORS[ranked["method"]], xerr=stds,
                error_kw={"ecolor": INK_2, "elinewidth": 0.8, "capsize": 0})
        ax.axvline(0, color=AXIS, linewidth=0.8)
        span = max(means.max(), 0) - min(means.min(), 0)
        for yi, (m, s) in enumerate(zip(means, stds)):
            # Past the bar's end or its error bar, whichever is further; a negative bar's goes right of 0.
            ax.text(max(m, 0) + s + 0.01 * span, yi, _number(m, ".3f"), va="center", fontsize=7.5, color=INK_2)
        ax.set_yticks(y, [f["feature"] for f in features], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(min(means.min(), 0) - 0.02 * span, max(means.max(), 0) + 0.12 * span)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"Fall in average precision when the feature is shuffled, mean ± SD of {ranked['n_repeats']}")
        name = method_name(ranked["method"], metrics["methods"][ranked["method"]]["params"])
        ax.set_title(f"Permutation importance: {name.lower()}, test sessions", loc="left", fontsize=10)
    return fig


# --- I/O ------------------------------------------------------------------------

def _write_text(text, path):
    """text to path, written aside and renamed into place, as run.py writes its files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(text)
    partial.replace(path)


def _save(fig, path):
    """fig to path as a PNG, written aside and renamed into place, without the matplotlib version in it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    fig.savefig(partial, format="png", dpi=DPI, facecolor=SURFACE, metadata={"Software": None})
    partial.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m spinwarn.report",
        description="Write the results table, the plots and the README's results section from results/metrics.json.",
    )
    parser.parse_args(argv)
    if not METRICS_PATH.exists():
        parser.error(f"no {METRICS_PATH}: `python -m spinwarn.run final` writes it")
    metrics = json.loads(METRICS_PATH.read_text())
    # The README is checked first, so a missing marker writes nothing.
    try:
        readme = replace_block(README_PATH.read_text(), readme_block(metrics))
    except ValueError as e:
        parser.error(str(e))

    _write_text(results_table(metrics), RESULTS_TABLE_PATH)
    _save(recall_figure(metrics), RECALL_FA_PLOT_PATH)
    _save(importance_figure(metrics), IMPORTANCE_PLOT_PATH)
    _write_text(readme, README_PATH)
    print(f"from {METRICS_PATH}, wrote {RESULTS_TABLE_PATH}, {RECALL_FA_PLOT_PATH}, {IMPORTANCE_PLOT_PATH} and the "
          f"results section of {README_PATH}")


if __name__ == "__main__":
    main()
