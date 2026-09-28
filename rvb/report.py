"""Turn an evaluation result dict into a short Markdown scorecard."""
from __future__ import annotations


def _f(v, nd=3, pct=False):
    if v is None:
        return "–"
    if isinstance(v, str):
        return v
    return f"{100*v:.1f}%" if pct else f"{v:.{nd}f}"


def to_markdown(res: dict, title: str = "Evaluation report") -> str:
    L = [f"# {title}", ""]
    if "detection" in res:
        L += ["## Level 1 · Detection", "", "| Camera/session | AP50 | Precision | Recall | F1 | Recall small <50px |", "|---|---|---|---|---|---|"]
        for k, d in res["detection"].items():
            L.append(f"| {k} | {_f(d['AP50'])} | {_f(d['precision'])} | {_f(d['recall'])} | {_f(d['F1'])} | {_f(d['recall_by_size'].get('small <50px'))} |")
        L.append("")
    if "tracking" in res:
        c = res["tracking"]["combined"]
        L += ["## Level 2 · Tracking", "", f"Scope: {res['tracking'].get('scope', 'all people')}", "", "| Sequence | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |", "|---|---|---|---|---|---|---|"]
        for k, d in res["tracking"]["per_sequence"].items():
            L.append(f"| {k} | {_f(d['HOTA'])} | {_f(d['DetA'])} | {_f(d['AssA'])} | {_f(d['MOTA'])} | {_f(d['IDF1'])} | {int(d['IDSW'])} |")
        L.append(f"| **combined** | **{_f(c['HOTA'])}** | {_f(c['DetA'])} | {_f(c['AssA'])} | {_f(c['MOTA'])} | **{_f(c['IDF1'])}** | {int(c['IDSW'])} |")
        L.append("")
    if "reid" in res:
        r = res["reid"]
        L += ["## Level 2 · Re-ID", "", f"Rank-1 {_f(r.get('rank1'))} · Rank-5 {_f(r.get('rank5'))} · mAP {_f(r.get('mAP'))} "
              f"({r.get('n_valid_queries')} queries)", ""]
    if "business" in res:
        b = res["business"]
        f, z = b["footfall"], b["zones"]
        L += ["## Level 3 · Business answers", "",
              "| Metric | Result |", "|---|---|",
              f"| Footfall total (GT → pred) | {f['gt_total']} → {f['pred_total']} ({f['total_pct_error']:+.1f}%) |",
              f"| Footfall hourly MAE / MAPE | {_f(f['hourly_MAE'], 2)} people / {_f(f['hourly_MAPE'], 1)}% |",
              f"| Peak hours overlap@{f['peak_k']} / Spearman | {_f(f['peak_overlap_at_k'], 2)} / {_f(f['peak_spearman'], 2)} |",
              f"| Zone visits matched (recall / precision) | {_f(z['visit_recall'], pct=True)} / {_f(z['visit_precision'], pct=True)} |",
              f"| Stop-vs-pass F1 (matched visits) | {_f(z['stop_vs_pass']['F1'])} |",
              f"| Engagement-rate abs error (mean over zones) | {_f(z['engagement_rate_abs_error'], pct=True)} |",
              f"| Mean dwell abs error (engaged visits) | {_f(z['mean_dwell_abs_error_s'], 1)} s |",
              f"| Occupancy MAE per minute | {_f(z['occupancy_MAE_per_min'], 2)} people |", "",
              "| Zone | Visits GT→pred | Engaged GT→pred | Engagement rate GT→pred | Mean dwell GT→pred (s) |", "|---|---|---|---|---|"]
        for zid, d in z["per_zone"].items():
            L.append(f"| {zid} | {d['gt_visits']}→{d['pred_visits']} | {d['gt_engaged']}→{d['pred_engaged']} | "
                     f"{_f(d['gt_engagement_rate'], pct=True)}→{_f(d['pred_engagement_rate'], pct=True)} | "
                     f"{_f(d['gt_mean_dwell_s'], 1)}→{_f(d['pred_mean_dwell_s'], 1)} |")
        if b.get("transitions"):
            L += ["", "Zone-to-zone flow error (L1 of normalised matrices, 0 = identical, 2 = disjoint): " +
                  ", ".join(f"{k} {_f(v['L1_normalised'], 2)}" for k, v in b["transitions"].items())]
        L.append("")
    if "vlm" in res:
        v = res["vlm"]
        L += ["## Evidence / VLM", "", f"Evidence coverage {_f(v['evidence_coverage'], pct=True)} · grounding {_f(v['evidence_grounding'], pct=True)} · "
              f"numeric accuracy (±{int(100*v['numeric_tolerance'])}%) {_f(v['numeric_accuracy'], pct=True)} over {v['n_insights']} insights", ""]
    return "\n".join(L)
