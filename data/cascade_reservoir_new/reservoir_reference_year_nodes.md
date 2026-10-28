# Year-specific soft reference paths

`reservoir_reference_year_nodes.csv` explicitly defines the retained edited
storage paths for THR in 2006, 2011 and 2022, and XJB in 2022. These are supplied
soft references, with status
`retained_edited_soft_reference_not_verified_operation`; this reconciliation
does not establish their origin, optimality or authenticity.

The recurring nodes in `reservoir_reference_level_nodes.csv` and the default
period summaries in `reservoir_period_operation_parameters.csv` still apply
to all other reservoir/year pairs. The default period table continues to
define calendar periods, band widths, physical metadata and reporting weights
for the overridden years. Their year-specific period endpoints come from
the dated nodes instead of the recurring period summary.

Each override has `reservoir_id`, integer `year`, consecutive `node_sequence`,
ISO `date`, `reference_target_level_m`, `reference_target_storage_m3`, and
`value_status`. Nodes represent start-of-day boundaries, begin on January 1,
and include the following January 1 boundary. Interpolation is linear in
storage over actual calendar days. Both endpoints match that year's initial
storage; each node must satisfy the stage–storage curve and global operating
bounds. These four paths include every daily boundary to retain the edited
storage values. The daily reference CSV is checked against these independent
nodes; editing only one table is rejected. The override file is optional for
datasets containing only recurring references, and is hashed when loaded.

These nodes describe each calendar year's operating reference within a
multi-year episode. They do not reset the simulated reservoir state on
January 1. The standard episode contains 1,825 consecutive daily transitions;
storage carries across the yearly reference boundaries, and return to the
episode's initial storage is imposed only at the final episode transition.

The original edited daily CSV was backed up before reconciliation to
`runs/input_backups/seasonal_reference_20261025/`, with original SHA-256
`40655c6cd38856e857489f3a822cdbf72763549c47f4a39a38a7bb1b04057eba`.
The only changed storage boundary is XJB at the end of 2022-10-24 / start of
2022-10-25: 4,977,000,002 m³ was corrected to the configured maximum of
4,977,000,000 m³. All other storage targets in the edited file were retained.
Levels, daily storage changes/rates, subphase labels and soft bands were
reconciled with those targets. Subphases use the sign of storage change,
treating changes of at most 1 m³ as hold.

Both soft band endpoints are clipped to the interval between the operating
minimum and the whole-day reference ceiling. This prevents inverted bands
when an edited target exceeds a flood ceiling; it does not move the target
or raise that ceiling. `reference_target_feasible_zero_lag` is recomputed as
`0 <= cumulative_natural_inflow - cumulative(storage_change / 86400)
<= max_total_release`, with the loader's existing flow tolerance. This flag
checks release balance only. It does **not** certify hard seasonal compliance.

Hard flood ceilings and refill deadlines apply to every included calendar
year. Reference paths remain descriptive inputs and do not permit the
executed policy to violate a seasonal ceiling, refill deadline or the final
episode storage target.
