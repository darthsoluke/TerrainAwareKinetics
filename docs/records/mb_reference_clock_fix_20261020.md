# Reference Clock Boundary Repair

The user's v47 episode3 failed at14.2s while extending an existing reference.
The requested consumer window was[14.4,15.0]s. Its55real native samples covered
[13.199999999999996,14.999999999999996]s. A3.552713678800501e-15s roundoff mismatch
passed the scheduler's numerical comparison but failed the strict source-interval
constructor. The child traceback identifies `reference_stream.py`'s retarget
window construction. This was not missing generated motion.

The continuous stream now represents its native30Hz timestamps on integer150Hz
ticks, shared with50Hz control. Control endpoints use integer control ticks.
Retarget sampling also derives times from integer ticks, with source endpoints
represented exactly when only floating-point roundoff separates them. Actual
out-of-range requests still fail; no motion is extrapolated, held or synthesized.
All pose/position arrays remain unchanged by time normalization.

Regression includes the exact failed user's numeric artifact and its accepted
parent reference. The repaired call produces all31samples from14.4through15.0s
and preserves the existing prefix. A15.02s request without new source data is
still rejected.228continuation, retarget and runtime tests pass, including saved
legacy sources with floating-point endpoint phases.

Original failure evidence remains in
`interactive_session_ready_v47/recording/episode_0003_000.jsonl` and its
`episode_0003_numeric/a2dea30cdbd0a06b5415b747f557d42f3296c630bf8664ff9dc7cbb57eb843c7.npz`.
No training, quality-threshold change or physical-state replacement was used.

Package `interactive_clock_v48` preserves all15PFNN maps and the user's last
selected Urban scene. Native/compiled preparation and live000-map regression
are being checked before returning the normal control page. Server21148,
planner12884, port28948; explicit development GPU-power scope.
