Probe phase: measuring sensitivity. Each candidate changes exactly ONE mechanism starting from the official baseline (not from previous candidates) so its effect is attributable. Prefer mechanisms likely to help TNS on every circuit.

Probe menu (pick ones not yet tried in the history; earlier items first):
  P1  High-fan-out-robust cut term: replace 1-prod(p)-prod(1-p) by a star/pairwise expansion or logsumexp soft-range so large nets keep a gradient; or down-weight nets by 1/log(fanout).
  P2  Balance hinge: keep the quadratic term but add a steep penalty outside |ratio-0.5| > 0.01 (dead-band), so the raw partition lands inside the band and the repair step barely moves anything.
  P3  Criticality-weighted cut: net weight w_n = 1 + alpha * max_pin_crit where crit = (1 - slack_rank_quantile)^2 computed INSIDE the case; alpha ~ 2-4.
  P4  Critical-tier preference: push the most critical nodes (top quantile) toward ONE tier with a small linear term mu * sum(crit * p); tests whether the fixed repair flow is tier-asymmetric.
  P5  Macro balance: explicit hinge terms on upper-tier macro count ratio and macro area ratio (guard the zero-macro case).
  P6  Critical-pair Laplacian: sum over edges of crit_i*crit_j*(p_i-p_j)^2 so critical paths stay on one tier.
  P7  Critical-area balance: the criticality-weighted area should also be ~50/50 (soft hinge, dead-band 0.05).
  P8  Binarisation shape: replace p(1-p) by entropy or scale its coefficient; tests whether early binarisation locks in bad structure.
