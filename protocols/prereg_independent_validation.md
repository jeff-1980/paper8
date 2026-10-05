# Pre-registration: independent-bearing / independent-condition validation of the added-channel effect (BM3 paper, revision 2026-09-29)

Written before any of the runs below were started. Frozen; deviations are listed in the results summary.

## Data (all local)
XJTU-SY, 15 bearings; usable OR/IR labels (from bearmamba3/data_xjtu.py BEARING_FAILURE): Cond1 (2100 rpm): 1_1 OR, 1_2 OR, 1_3 OR; Cond2 (2250 rpm): 2_1 IR, 2_2 OR, 2_4 OR, 2_5 OR; Cond3 (2400 rpm): 3_1 OR, 3_3 IR, 3_4 IR, 3_5 OR. Excluded (mixed/cage labels): 1_4, 1_5, 2_3, 3_2. The main paper used only Cond2->Cond3 and Cond3 LOBO; Cond1 bearings and Cond2 as a test condition were never evaluated.

## Fold definitions (leave-one-condition-out, LOCO)
- L2: test Cond2 (2_1 IR; 2_2, 2_4, 2_5 OR); train Cond1+Cond3 (OR: 1_1,1_2,1_3,3_1,3_5; IR: 3_3,3_4).
- L1: test Cond1 (1_1,1_2,1_3, all OR); train Cond2+Cond3 (OR: 2_2,2_4,2_5,3_1,3_5; IR: 2_1,3_3,3_4).
- L3: test Cond3 (3_1 OR, 3_3 IR, 3_4 IR, 3_5 OR); train Cond1+Cond2 (OR: 1_1,1_2,1_3,2_2,2_4,2_5; IR: 2_1).
Arms: H (horizontal), V (vertical), HV (dual); backbone BM3 (primary), 1D-CNN (secondary). Same hyperparameters as the main paper; fixed final epoch; no test-based selection; no hyperparameter tuning. Seeds 0-2 (BM3) and 0-4 (1D-CNN).

## OR same-class coverage sweep
Fixed test bearing 3_1 (OR). Training pool always contains the three IR bearings 2_1, 3_3, 3_4; OR training bearings added in this order: k=1: 3_5; k=2: +2_2; k=3: +2_4; k=4: +2_5; k=5: +1_1; k=6: +1_2; k=7: +1_3. Arms H and HV (and V if time allows), seeds 0-2. Caveat stated in advance: k>1 adds bearings from other operating conditions, so coverage is confounded with condition diversity.

## Metrics and unit of analysis
Primary metric per (fold, test bearing): recall of the bearing's own class (all windows of a test bearing have one label). Seed-level means are computed within bearing. The unit of inference for statements about bearings is the test bearing; seeds only quantify training stochasticity and are never used alone to claim generalisation across bearings.

## Pre-specified contrasts
C1: HV minus V (does adding the horizontal channel help over the vertical channel alone), C2: HV minus H, C3: V minus H, each computed per test bearing (mean over seeds) and summarised by the number of test bearings with positive difference out of the total, and the mean per-bearing difference with a bootstrap over bearings (bootstrap CIs are descriptive given <=11 bearings). Hypotheses stated in advance: (i) if the fusion-specific benefit exists, C1 and C2 are positive in a majority of test bearings; (ii) if the vertical channel alone accounts for the gain, C3 is positive and C1 is near zero or negative; (iii) the outer-race degradation of HV relative to H seen in Cond3 LOBO is reproduced if C2 is negative for most OR test bearings in L2 and L1. No claim will be made for a contrast whose sign is inconsistent across bearings.
## Reporting rule
All arms x folds x seeds run are reported, including failures and collapsed models; a model that predicts one class for all windows is flagged as collapsed.
