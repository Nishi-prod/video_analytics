# Demo - simulated 'medium' tracker (synthetic data)

## Level 1 · Detection

| Camera/session | AP50 | Precision | Recall | F1 | Recall small <50px |
|---|---|---|---|---|---|
| cam_entrance/wkday_mon | 0.838 | 0.967 | 0.847 | 0.903 | – |
| cam_entrance/wkend_sat | 0.850 | 0.970 | 0.853 | 0.908 | – |
| cam_iphone/wkday_mon | 0.853 | 0.975 | 0.855 | 0.911 | – |
| cam_iphone/wkend_sat | 0.838 | 0.976 | 0.839 | 0.902 | – |
| cam_samsung/wkday_mon | 0.849 | 0.975 | 0.852 | 0.909 | – |
| cam_samsung/wkend_sat | 0.844 | 0.975 | 0.845 | 0.906 | – |

## Level 2 · Tracking

Scope: customers (staff boxes ignored)

| Sequence | HOTA | DetA | AssA | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| cam_entrance/wkday_mon | 0.578 | 0.615 | 0.545 | 0.808 | 0.783 | 20 |
| cam_entrance/wkend_sat | 0.575 | 0.625 | 0.530 | 0.817 | 0.762 | 38 |
| cam_iphone/wkday_mon | 0.468 | 0.560 | 0.392 | 0.718 | 0.609 | 226 |
| cam_iphone/wkend_sat | 0.448 | 0.554 | 0.362 | 0.697 | 0.586 | 641 |
| cam_samsung/wkday_mon | 0.453 | 0.540 | 0.380 | 0.678 | 0.565 | 261 |
| cam_samsung/wkend_sat | 0.429 | 0.552 | 0.333 | 0.694 | 0.558 | 593 |
| **combined** | **0.456** | 0.556 | 0.373 | 0.704 | **0.590** | 1779 |

## Level 3 · Business answers

| Metric | Result |
|---|---|
| Footfall total (GT → pred) | 124 → 113 (-8.9%) |
| Footfall hourly MAE / MAPE | 2.75 people / 9.2% |
| Peak hours overlap@3 / Spearman | 1.00 / 1.00 |
| Zone visits matched (recall / precision) | 88.5% / 73.5% |
| Stop-vs-pass F1 (matched visits) | 0.952 |
| Engagement-rate abs error (mean over zones) | 5.8% |
| Mean dwell abs error (engaged visits) | 5.0 s |
| Occupancy MAE per minute | 0.04 people |

| Zone | Visits GT→pred | Engaged GT→pred | Engagement rate GT→pred | Mean dwell GT→pred (s) |
|---|---|---|---|---|
| iphone_table | 114→142 | 49→54 | 43.0%→38.0% | 57.7→44.7 |
| iphone_wall | 44→57 | 20→20 | 45.5%→35.1% | 43.0→42.7 |
| samsung_table | 88→104 | 33→34 | 37.5%→32.7% | 31.6→27.6 |
| samsung_wall | 33→33 | 12→13 | 36.4%→39.4% | 31.4→28.8 |

Zone-to-zone flow error (L1 of normalised matrices, 0 = identical, 2 = disjoint): cam_iphone 0.05, cam_samsung 0.08
