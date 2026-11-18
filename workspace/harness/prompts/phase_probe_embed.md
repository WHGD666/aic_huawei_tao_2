Probe phase: measuring sensitivity. Each candidate changes exactly ONE mechanism starting from the official baseline (not from previous candidates) so its effect is attributable. You may change the loss OR the decoder in a given candidate, not both.

Probe menu (pick ones not yet tried in the history; earlier items first):
  D1  Decoder: after clustering, run a criticality-weighted Fiduccia–Mattheyses-style refinement (gain = weighted cut change, weights 1+alpha*crit) under the balance constraint (area 49-51%, macro count/area 46-54%), vectorised with numpy, bounded passes.
  D2  Decoder: balance correction flips the LEAST critical / lowest-degree nodes first instead of nearest-to-other-centroid.
  D3  Decoder: assign macros first (small exhaustive / greedy search satisfying 45-55% count and area), then cells.
  L1  Loss: weight positive pairs by net criticality (1 + alpha*max(crit_i, crit_j)) so critical neighbours embed closer.
  L2  Loss: normalise positive pairs by net fan-out (clique weight 1/(k-1)) so hub nets do not collapse everything.
  L3  Loss: temperature / negative weight (pure coefficient probe).
  L4  Loss: add 2-hop critical positives (nodes two hops apart on a critical path pulled together).
