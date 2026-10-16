"""
Tests for spinwarn.report: the cells, the paired comparison, the README's markers, the curves and the command.

A hand-made metrics.json with the keys report reads, so they run anywhere
without results/. The last two check the repo's own files against the
committed results/metrics.json, and skip without it.
"""

import json

import pytest

from spinwarn import report
from spinwarn.config import METRICS_PATH, MIN_KIND_SPINS, README_PATH, RESULTS_TABLE_PATH


SPINS = [("a", 5.0), ("a", 20.0), ("b", 7.5), ("b", 30.0)]
KINDS = {"a": "natural", "b": "deliberate"}
# A curve, thresholds shuffled: always on at 0 (every frame alarms, 1.1 FA/min,
# every spin), most false alarms at 1, and 3 off the curve, since 2 is lower
# and gives fewer.
CURVE = {"threshold": [3, 0, 5, 1, 4, 2], "fa_per_min": [4.0, 1.1, 0.0, 6.0, 0.8, 3.0],
         "recall_min_lead": [0.95, 1.0, 0.0, 1.0, 0.5, 0.9], "recall": [1.0, 1.0, 0.0, 1.0, 0.75, 1.0],
         "time_in_alarm": [0.1, 1.0, 0.0, 0.5, 0.02, 0.2]}


def summary(spins, fa_per_min):
    """summarize()'s keys for spins, with the intervals operating_point adds."""
    n = len(spins)
    caught = sum(s["caught"] for s in spins)
    headline = sum(s["caught_min_lead"] for s in spins)
    return {"n_spins": n, "caught": caught, "recall": caught / n if n else None, "caught_min_lead": headline,
            "recall_min_lead": headline / n if n else None, "alerts": caught + 2, "false_alarms": 2,
            "minutes": 2.0, "fa_per_min": fa_per_min, "alert_precision": caught / (caught + 2),
            "time_in_alarm": 0.0215, "median_lead": 0.6 if caught else None,
            "recall_min_lead_ci": [0.25, 1.0] if n else [None, None], "recall_ci": [0.5, 1.0] if n else [None, None]}


def point(target, headline, any_lead=None, fa_per_min=1.0):
    """A test operating point with one bool per spin of SPINS caught with the headline's lead, and with any."""
    any_lead = headline if any_lead is None else any_lead
    spins = [{"session": s, "t": t, "caught": c, "caught_min_lead": h, "lead": (0.6 if h else 0.3) if c else None,
              "kind": KINDS[s]} for (s, t), h, c in zip(SPINS, headline, any_lead)]
    by_kind = {kind: {**summary([s for s in spins if s["kind"] == kind], fa_per_min),
                      "enough": sum(s["kind"] == kind for s in spins) >= MIN_KIND_SPINS}
               for kind in sorted(KINDS.values())}
    return {"fa_target": target, "threshold": 0.5, **summary(spins, fa_per_min), "by_kind": by_kind, "spins": spins}


def method(params, at_1, at_half, timing=None, chosen_by="nothing to tune"):
    """A method's entry: CV with CURVE, and test at 1 and 0.5 FA/min with the headline catches given."""
    return {
        "run_id": "run", "params": params, "chosen_by": chosen_by,
        "cv": {"pr_auc": 0.8, "operating_points": [], "curve": CURVE},
        "test": {"pr_auc": 0.75, "operating_points": [point(1.0, at_1, [True] * 4, 1.1), point(0.5, at_half)]},
        "timing": timing or {"ms": 0.005125},
    }


@pytest.fixture
def metrics():
    caught, missed = [True, True, True, False], [False, True, False, False]
    return {
        "started": "2026-10-16T17:50:58", "horizon": 0.5, "forced": False, "min_lead_s": 0.5,
        "min_kind_spins": MIN_KIND_SPINS,
        "dev": {"sessions": ["c", "d", "e"], "spins": 30, "scored_min": 25.04},
        "test": {"sessions": {"a": {"kind": "natural", "spins": 2, "scored_min": 2.0},
                              "b": {"kind": "deliberate", "spins": 2, "scored_min": 2.0}},
                 "spins": 4, "scored_min": 4.0},
        "methods": {
            "b1": method({}, missed, missed),
            "b2": method({"theta1": 25}, missed, missed),
            "b3": method({}, caught, missed),
            "lr": method({"C": 0.01}, caught, caught, {"ms": 0.07517}, "PR-AUC, best of 4 configs"),
            "rf": method({"max_depth": None}, caught, caught, {"ms": 13.234, "ms_n_jobs_1": 3.352}),
            "hgb": method({"learning_rate": 0.1}, [True] * 4, caught, {"ms": 1.532}),
        },
        "importance": {"method": "hgb", "chosen_by": "CV PR-AUC, best of the models", "scoring": "average_precision",
                       "n_repeats": 10, "features": [{"feature": "time_to_90_last", "mean": 0.7, "std": 0.005},
                                                     {"feature": "brake_mean", "mean": -0.006, "std": 0.001}]},
    }


# --- cells ------------------------------------------------------------------------

def test_recall_cell_has_the_recall_the_count_and_the_interval():
    p = point(1.0, [True, True, True, False], [True] * 4)
    assert report.recall_cell(p) == "0.75 (3/4) [0.25, 1.00]"
    assert report.recall_cell(p, headline=False) == "1.00 (4/4) [0.50, 1.00]"


def test_recall_cell_with_no_spins_has_a_dash_and_no_interval():
    assert report.recall_cell(summary([], 0.0)) == "— (0/0)"


def test_ms_cell_is_3_figures_with_one_core_where_timed():
    assert report.ms_cell({"ms": 0.0051249}) == "0.00512"
    assert report.ms_cell({"ms": 13.234, "ms_n_jobs_1": 3.352}) == "13.2 (3.35 on one core)"


@pytest.mark.parametrize("x, spec, text", [
    (-0.006, ".3f", "−0.006"), (-3.4e-5, ".3f", "0.000"), (-0.0, ".2f", "0.00"), (-0.004, "+.2f", "+0.00"),
    (-1e-5, ".1%", "0.0%"), (-0.04, "+.2f", "−0.04"), (None, ".2f", "—"),
])
def test_a_number_that_rounds_to_0_has_no_minus_sign(x, spec, text):
    assert report._number(x, spec) == text


def test_method_name_gives_b2_its_theta1():
    assert report.method_name("b2", {"theta1": 25}) == "B2: slip rate, |slip| > 25°"
    assert report.method_name("rf", {"max_depth": None}) == "Random forest"


# --- the paired comparison ------------------------------------------------------

def test_paired_difference_of_a_method_with_itself_is_zero_with_no_width():
    # Each resample draws the same spins for both, so the difference never
    # moves; resampled apart, the two recalls would.
    caught = [True] * 14 + [False] * 6
    assert report.paired_difference(caught, caught) == (0.0, 0.0, 0.0)


def test_paired_difference_is_the_same_on_every_run_with_the_same_seed():
    a = [True] * 20 + [False] * 7
    b = [True] * 15 + [False] * 5 + [True] * 2 + [False] * 5
    difference, lo, hi = report.paired_difference(a, b)
    assert report.paired_difference(a, b) == (difference, lo, hi)
    assert difference == pytest.approx(3 / 27)
    assert lo < difference < hi


def test_paired_difference_of_methods_that_split_the_spins_spans_zero():
    a = [True, False] * 10
    b = [False, True] * 10
    difference, lo, hi = report.paired_difference(a, b)
    assert difference == 0.0
    assert lo < 0 < hi
    assert report.paired_difference([True] * 9, [False] * 9) == (1.0, 1.0, 1.0)


def test_paired_difference_refuses_mismatched_or_empty_catches():
    with pytest.raises(ValueError):
        report.paired_difference([True, False], [True])
    with pytest.raises(ValueError):
        report.paired_difference([], [])


def test_paired_counts_the_spins_only_one_catches():
    model = point(1.0, [True, True, False, True])
    baseline = point(1.0, [True, False, True, False])
    c = report.paired(model, baseline)
    assert (c["only_model"], c["only_baseline"]) == (2, 1)
    assert c["difference"] == pytest.approx(0.25)


def test_paired_refuses_operating_points_with_other_spins():
    model = point(1.0, [True] * 4)
    baseline = point(1.0, [True] * 4)
    baseline["spins"][2]["t"] = 8.0
    with pytest.raises(ValueError, match="different spins"):
        report.paired(model, baseline)


# --- the tables -----------------------------------------------------------------

def test_results_table_has_every_method_at_every_threshold(metrics):
    text = report.results_table(metrics)
    assert text.startswith("# Results\n")
    assert "## At the 1 FA/min threshold" in text and "## At the 0.5 FA/min threshold" in text
    # B2's name, with its | escaped for the table, at 1 FA/min on test.
    assert "| B2: slip rate, \\|slip\\| > 25° | 0.25 (1/4) [0.25, 1.00] | 1.00 (4/4) [0.50, 1.00] | 1.10 |" in text
    assert "one test spin is 25.0 points of recall" in text
    assert "| Random forest | `{\"max_depth\": null}` |" in text
    assert "13.2 (3.35 on one core)" in text


def test_kinds_with_too_few_spins_say_so(metrics):
    text = "\n".join(report.kinds_table(metrics))
    assert f"natural (2, under {MIN_KIND_SPINS})" in text


def test_paired_table_has_each_model_against_b3_then_b2(metrics):
    rows = report.paired_table(metrics)[2:]
    assert len(rows) == 6
    assert rows[0].startswith("| Logistic regression | B3: time to 90° | 0 / 0 | +0.00 [0.00, 0.00] | 2 / 0 |")
    assert rows[5].startswith("| Gradient boosting | B2: slip rate, \\|slip\\| > 25° | 3 / 0 | +0.75")


def test_a_forced_run_says_so(metrics):
    assert "scored the test sessions once" in report.intro(metrics)
    metrics["forced"] = True
    assert "forced: the test sessions had been scored before" in report.intro(metrics)


def test_readme_block_links_both_plots_from_the_readme(metrics):
    block = report.readme_block(metrics)
    assert "](results/recall_vs_fa.png)" in block
    assert "](results/importance.png)" in block
    assert "### At the 1 FA/min threshold" in block


# --- the README's markers -------------------------------------------------------

def test_replace_block_rewrites_only_between_the_markers():
    text = f"# Title\n\nStory.\n\n{report.RESULTS_START}\nold\n{report.RESULTS_END}\n\n## Limitations\n"
    new = report.replace_block(text, "new table")
    assert new == f"# Title\n\nStory.\n\n{report.RESULTS_START}\nnew table\n{report.RESULTS_END}\n\n## Limitations\n"
    assert report.replace_block(new, "new table") == new


@pytest.mark.parametrize("text", [
    "no markers",
    f"{report.RESULTS_START}\nonly the start",
    f"{report.RESULTS_END}\n{report.RESULTS_START}",
    f"{report.RESULTS_START}\n{report.RESULTS_END}\n{report.RESULTS_START}\n{report.RESULTS_END}",
])
def test_replace_block_refuses_a_readme_without_one_pair_of_markers_in_order(text):
    with pytest.raises(ValueError):
        report.replace_block(text, "table")


# --- the curves -----------------------------------------------------------------

def test_curve_steps_starts_at_the_most_false_alarms_and_keeps_threshold_for_fas_picks():
    fa, recall = report.curve_steps(CURVE, "recall_min_lead")
    # Thresholds 5, 4, 2 and 1; 0 is the alarm left on, and 3 gives more than 2.
    assert fa.tolist() == [0.0, 0.8, 3.0, 6.0]
    assert recall.tolist() == [0.0, 0.5, 0.9, 1.0]


@pytest.mark.parametrize("rate, recall", [(0.0, 0.0), (0.5, 0.0), (1.0, 0.5), (3.0, 0.9), (3.5, 0.9), (10.0, 1.0)])
def test_recall_at_is_the_recall_of_the_lowest_threshold_within_the_rate(rate, recall):
    # At 3.5 the lowest threshold within the rate is 0, every frame alarmed,
    # under the warning range: threshold_for_fa sets 2, and so does the curve.
    assert report.recall_at(CURVE, "recall_min_lead", rate) == recall


# --- the command ----------------------------------------------------------------

@pytest.fixture
def paths(tmp_path, monkeypatch):
    """report's paths in tmp_path, with a README that has the markers."""
    paths = {"METRICS_PATH": tmp_path / "results" / "metrics.json", "README_PATH": tmp_path / "README.md",
             "RESULTS_TABLE_PATH": tmp_path / "results" / "results_table.md",
             "RECALL_FA_PLOT_PATH": tmp_path / "results" / "recall_vs_fa.png",
             "IMPORTANCE_PLOT_PATH": tmp_path / "results" / "importance.png"}
    for name, path in paths.items():
        monkeypatch.setattr(report, name, path)
    paths["METRICS_PATH"].parent.mkdir()
    paths["README_PATH"].write_text(f"# spin\n\n{report.RESULTS_START}\n{report.RESULTS_END}\n\n## Limitations\n")
    return paths


def test_main_writes_the_table_the_plots_and_the_readme(paths, metrics):
    paths["METRICS_PATH"].write_text(json.dumps(metrics))
    report.main([])
    assert paths["RESULTS_TABLE_PATH"].read_text() == report.results_table(metrics)
    for plot in ("RECALL_FA_PLOT_PATH", "IMPORTANCE_PLOT_PATH"):
        assert paths[plot].read_bytes().startswith(b"\x89PNG")
    readme = paths["README_PATH"].read_text()
    assert readme.startswith("# spin\n\n") and readme.endswith(f"{report.RESULTS_END}\n\n## Limitations\n")
    assert report.readme_block(metrics) in readme
    assert not list(paths["METRICS_PATH"].parent.glob("*.partial"))


def test_main_refuses_a_readme_without_markers_and_writes_nothing(paths, metrics):
    paths["METRICS_PATH"].write_text(json.dumps(metrics))
    paths["README_PATH"].write_text("# spin\n")
    with pytest.raises(SystemExit):
        report.main([])
    assert paths["README_PATH"].read_text() == "# spin\n"
    assert not paths["RESULTS_TABLE_PATH"].exists() and not paths["RECALL_FA_PLOT_PATH"].exists()


def test_main_refuses_without_metrics(paths):
    with pytest.raises(SystemExit):
        report.main([])


# --- the repo's own files ---------------------------------------------------------

needs_metrics = pytest.mark.skipif(not METRICS_PATH.exists(), reason="no results/metrics.json")


@needs_metrics
def test_the_readme_and_results_table_are_what_report_writes_from_metrics_json():
    metrics = json.loads(METRICS_PATH.read_text())
    readme = README_PATH.read_text()
    assert report.replace_block(readme, report.readme_block(metrics)) == readme, "run python -m spinwarn.report"
    assert RESULTS_TABLE_PATH.read_text() == report.results_table(metrics), "run python -m spinwarn.report"


@needs_metrics
def test_every_cv_curve_gives_its_methods_operating_points():
    # The plot and the final run's thresholds agree at both targets.
    for name, result in json.loads(METRICS_PATH.read_text())["methods"].items():
        for p in result["cv"]["operating_points"]:
            for key in ("recall_min_lead", "recall"):
                assert report.recall_at(result["cv"]["curve"], key, p["fa_target"]) == p[key], (name, key)
