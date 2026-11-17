# Battery swapping station planning for inland electric ships

Code and parameters behind the paper *Flexible and Differential Battery
Swapping for Inland Battery Electric Ships: Station Siting, Initial Inventory
Sizing and Benefit Assessment* (eTransportation, manuscript ETRAN-D-26-00296).

This repository is published so that a reader can see how each result was
computed. It is a curated copy of the internal working tree, with the internal
quality gates, the figure scripts and the writing toolchain left out.

## What is here

| Path | Paper section | What it does |
|---|---|---|
| `src/profile.py` | 3.2.2 | Lateral velocity profile, Equations (1) and (2) |
| `src/geometry.py` | 3.2.3 | Normal projection and the lateral coordinate, Equations (3) and (4) |
| `s2_flowfield/run.py` | 3.2 | Builds the flow field over the corridor |
| `1_pipeline/calibrate_flow_field.py` | 3.2.2, 4.1.3 | Anchors the section-mean velocity on the gauge record |
| `1_pipeline/rebuild_flow_velocities.py` | 3.2.4 | Section-mean velocity per leg |
| `src/metrics.py` | 3.3.4 | Admiralty power and leg energy, Equations (5) to (8) |
| `1_pipeline/calibrate_admiralty_pership.py` | 3.3.4, 4.1.3 | Coefficient per ship from the power regression |
| `s3_energy/run.py` | 3.3 | Energy for every voyage leg |
| `2_analysis/build_soc.py` | 3.4.2 | State of charge per leg, Equations (15a) and (15b) |
| `1_pipeline/resolve_siting.py` | 3.4 | Fixed-size maximum coverage over station combinations |
| `s4_siting/run.py` | 3.4, 4.2, 4.3.1 | Coverage and the selected stations |
| `1_pipeline/rebuild_inventory_chain.py` | 3.5.3, 3.5.4 | Daily station inventory and opening stock, Equations (22) to (27) |
| `src/timechain.py` | 4.3.2 | Port time chain from arrival to departure |
| `2_analysis/build_transfer_model.py` | 3.5, 4.4 | Inter-port transfer inside the plan |
| `s5_inventory/run.py` | 3.5, 4.4 | Opening inventory under both swapping modes |
| `2_analysis/build_lcc.py` | 3.6, Appendix G | Annualized cost items and their inputs |
| `s6_cost/run.py` | 3.6, 4.3.3 | Cost per kilometer for the four plans |
| `2_analysis/build_uncertainty.py` | 4.5.1 | Perturbed runs, fixed seed |
| `2_analysis/build_queueing.py` | 4.5.2 | Service capacity and waiting |
| `s7_robustness/run.py` | 4.5 | Robustness of the planning solution |
| `2_analysis/build_pack_soc.py` | 3.5.2, 5.4 | Whole-pack replay, Equations (21a) to (21d), and the effect of battery mass |
| `2_analysis/siting_robustness_and_mass.py` | 4.2, 5.4, Appendix D | Eight-station layouts across energy density, and the full battery mass bound |
| `2_analysis/build_station_cost_sensitivity.py` | 5.1, Appendix F | Energy and reference station cost by station count |
| `2_analysis/station_count_extended.py` | 5.1, Appendix F | The same cost from four to twelve stations |
| `2_analysis/build_charging_replay.py` | 4.5.2 | Charging replay at each station from the observed arrival times |
| `2_analysis/build_coupled_inventory.py` | 4.5.2, Table G6 | Finite charging power, whole packs and opening charged packs |
| `2_analysis/build_rule_based_transfer.py` | 4.4, Table G5 | Transfer rule that uses only demand known on the day |
| `2_analysis/rolling_horizon_transfer.py` | 4.4, Table G5 | Rolling-horizon transfer with forecast or exact demand |
| `2_analysis/rule_transfer_allocation.py` | 4.4, 5.3 | Opening stock allocation and lower bounds for the transfer cases |
| `2_analysis/build_service_cost.py` | 3.6, 5.3 | Annualized station, battery and transport cost at equal service |
| `2_analysis/system_cost_accounts.py` | 3.6, 5.3, Tables G3 and G4 | Shipowner and system accounts, and the break-even values |
| `s8_release/run.py` | 4.1.3 | Builds the shared voyage-level table |
| `params.yaml` | Appendix A, Appendix G | Every parameter with its value and its source |

## Data

`data/voyages.csv` is the voyage-level table the study runs on. One row per
leg, with a stable code in place of the vessel identity and English port
names. The table is pseudonymized rather than anonymous. A reader with access
to commercial AIS could in principle match a route back to a vessel.

The raw AIS records are licensed from a commercial provider and are not
shared in any form. Appendix A of the paper reports every preprocessing
parameter and every exclusion criterion, so the screening can be followed
without the raw records.

## Parameters

`params.yaml` is the single source for every constant. Each entry carries its
value and where the value comes from. Entries marked `assumed` have no citable
source and are swept in the sensitivity analysis rather than presented as
measured.

## Reading order

Start at `params.yaml`, then follow the table above from Section 3.2 downward.
Each `run.py` is the entry point for one section of the paper and loads the
modules listed beside it.
