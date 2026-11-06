import sys, json, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from rclimpy.common import daily_frame_from_ymd, ANNUAL
from rclimpy.parameters import IndexParameters
from rclimpy.pipeline import compute_indices
from rclimpy.trend_analysis import ols_trend
raw = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/sampledataindcal.csv")
daily = daily_frame_from_ymd(raw)
params = IndexParameters(base_start=1915, base_end=2009, latitude=0.0, longitude=0.0, su_threshold=25, id_threshold=0, tr_threshold=20, fd_threshold=0, rnn_threshold=25)
ref = json.load(open("validation/reference_trend_stats.json")); ref.pop("_provenance")
t = time.time()
# annual only (monthly percentile bootstrap would double the run time); status of every index is kept
names = [n for n in ref] + ["TR20", "TNx", "TNn"]
from rclimpy.pipeline import build_specs
out = {}
for s in build_specs(params):
    if s.name in names:
        out[s.name] = s.compute(daily, params, ANNUAL)
print("compute sec", round(time.time() - t))
rows = []
for n, v in ref.items():
    st = ols_trend(out[n])
    got = [round(st["r2_pct"], 1), round(st["p_value"], 3), round(st["slope"], 3), round(st["slope_se"], 3)]
    rows.append(dict(index=n, ref_r2=v[0], our_r2=got[0], ref_p=v[1], our_p=got[1], ref_slope=v[2], our_slope=got[2], ref_se=v[3], our_se=got[3],
                     match=all(abs(a - b) < 1e-9 for a, b in zip(v, got)), n_years=st["n"],
                     exact_slope=round(st["slope"], 5)))
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.to_string(index=False))
print("exact matches:", int(df.match.sum()), "of", len(df))
df.to_csv("validation/trend_stats_comparison.csv", index=False)
pd.DataFrame({k: v for k, v in out.items()}).to_csv("validation/annual_indices_python.csv")
