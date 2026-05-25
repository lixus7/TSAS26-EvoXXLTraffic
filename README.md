<div align="center">
  <h2><b>EvoXXLTraffic: Evolving-Graph Traffic Forecasting under Extreme Sensor Growth</b></h2>
</div>

This directory accompanies the **TSAS'26** submission (`TSAS26_EVOXXLTraffic_Du_Yin/`) and documents the dataset, task, and baseline organisation that backs the main experimental tables ([`tables/tsas_main_table_part1.tex`](tables/tsas_main_table_part1.tex), [`tables/tsas_main_table_part2.tex`](tables/tsas_main_table_part2.tex)). All baselines below are implemented in [`eac/`](eac).

---

## 📊 Dataset and Task

We benchmark continual spatio-temporal forecasting on **EvoXXLTraffic** — a long-horizon extension of the PEMS family that spans up to **25 years** per district and exhibits **sensor-count growth of up to $+9,433\%$** between the first and last year. The task is *evolving-graph traffic forecasting*: at each period $\tau$ the sensor set $\mathcal{V}_\tau$ may expand (newly installed sensors) and the underlying graph $\mathcal{G}_\tau$ grows, while the model must keep predicting the next $3 / 6 / 12$ steps.

<p align="center">
    <img src="fig/new1.svg" alt="EvoXXLTraffic dataset and evolving-graph task overview" align="center" width="900px" />
</p>

> 📄 PDF version: [`fig/new1.pdf`](fig/new1.pdf) · prototype diagram: [`fig/proto.pdf`](fig/proto.pdf)

---

LaTeX source: [`table/data1.tex`](table/data1.tex).

### XXLTraffic subset from SIGSPATIAL 2025 Best Paper:

Comparison with existing traffic datasets:

| Reference | Dataset | Samples | Nodes | Interval | Span | Period |
|---|---|---:|---:|---|---|---|
| DCRNN | METR-LA | 34,272 | 207 | 5 min | 4 mo | 03/2012–06/2012 |
| DCRNN | PEMS-BAY | 52,116 | 325 | 5 min | 6 mo | 01/2017–05/2017 |
| LSTNet | Traffic | 17,544 | 862 | 1 h | 2 yr | 01/2015–12/2016 |
| STSGCN | PEMS03 | 26,208 | 358 | 5 min | 11 mo | 01/2018–11/2018 |
| STSGCN | PEMS04 | 16,992 | 307 | 5 min | 2 mo | 01/2018–02/2018 |
| STSGCN | PEMS07 | 28,224 | 883 | 5 min | 2 mo | 05/2017–06/2017 |
| STSGCN | PEMS08 | 17,856 | 170 | 5 min | 2 mo | 07/2016–08/2016 |
| Large-ST | CA / GLA / GBA / SD | 525,888 | 716 – 8,600 | 5 min | 5 yr | 01/2017–12/2021 |
| **Ours** | **PEMS03**<sub>gap&agg</sub> | 2,629,513 | 151 | Gap/Hr/Day | **23.00 yr** | 03/2001–03/2024 |
| **Ours** | **PEMS04**<sub>gap&agg</sub> | 2,486,472 | 822 | Gap/Hr/Day | **21.75 yr** | 06/2002–03/2024 |
| **Ours** | **PEMS05**<sub>gap&agg</sub> | 1,371,879 | 103 | Gap/Hr/Day | **12.00 yr** | 03/2012–03/2024 |
| **Ours** | **PEMS06**<sub>gap&agg</sub> | 1,628,852 | 130 | Gap/Hr/Day | **14.25 yr** | 12/2009–03/2024 |
| **Ours** | **PEMS07**<sub>gap&agg</sub> | 2,486,472 | 3,062 | Gap/Hr/Day | **21.75 yr** | 06/2002–03/2024 |
| **Ours** | **PEMS08**<sub>gap&agg</sub> | 2,629,513 | 212 | Gap/Hr/Day | **23.00 yr** | 03/2001–03/2024 |
| **Ours** | **PEMS10**<sub>gap&agg</sub> | 1,914,982 | 107 | Gap/Hr/Day | **16.75 yr** | 06/2007–03/2024 |
| **Ours** | **PEMS11**<sub>gap&agg</sub> | 2,457,676 | 521 | Gap/Hr/Day | **21.50 yr** | 09/2002–03/2024 |
| **Ours** | **PEMS12**<sub>gap&agg</sub> | 2,533,735 | 1,543 | Gap/Hr/Day | **22.16 yr** | 01/2002–03/2024 |

---

### EvoXXLTraffic subset from TSAS26-EvoXXLTraffic

Per-district sensor growth:

| District | Years | $N_\text{first}$ | $N_\text{last}$ | Growth |
|---|---|---:|---:|---:|
| PEMS03 | 2001–2025 (25) | 174 | 1,859 | $+968\%$ |
| PEMS04 | 2001–2025 (25) | $\sim 25$ | 4,089 | $\gg 10,000\%$ |
| PEMS05 | 2005–2025 (21) | $\sim 6$ | 573 | $\sim +9,433\%$ |
| PEMS06 | 2005–2025 (21) | $\sim 12$ | 705 | $\sim +5,638\%$ |
| PEMS07 | 2001–2025 (25) | $\sim 70$ | 4,888 | $\sim +6,883\%$ |
| PEMS08 | 2001–2025 (25) | $\sim 170$ | 2,059 | $\sim +1,111\%$ |
| PEMS10 | 2006–2025 (20) | $\sim 340$ | 1,378 | $\sim +305\%$ |
| PEMS11 | 1999–2025 (27) | $\sim 200$ | 1,440 | $\sim +620\%$ |
| PEMS12 | 2002–2025 (24) | $\sim 100$ | 2,587 | $\sim +2,487\%$ |

This regime (high growth $\times$ long horizon) is what existing evolving-graph methods are *not* designed for — backbones trained on the tiny first-year graph become severely under-capacity, and rank-limited prompts/embeddings cannot absorb the heterogeneity of thousands of newly installed sensors. EvoXXLTraffic is constructed precisely to expose this failure mode.

---

## 🛠️ Dataset Processing

All preprocessing notebooks live in [`xxltrafficdata/`](xxltrafficdata). Each district is processed by a two-stage pipeline:

```
raw PEMS dumps  ──[stage 1]──>  yearly per-district tensors  ──[stage 2]──>  EAC-format (flow + adj per year)
                pemsXX_yearly_nodes.ipynb                    pemsXX_build_eac_data.ipynb
```

* **Stage 1 — `pemsXX_yearly_nodes.ipynb`** ([example: PEMS03](xxltrafficdata/pems03_yearly_nodes.ipynb)) reads the raw 5-minute PEMS feed, harmonises the station set year-by-year, and emits yearly node lists / sensor metadata.
* **Stage 2 — `pemsXX_build_eac_data.ipynb`** ([example: PEMS03](xxltrafficdata/pems03_build_eac_data.ipynb)) takes the yearly node lists and produces the `<year>.npz` (flow tensor) and `<year>_adj.npz` (adjacency) files expected by [`eac/main.py`](eac/main.py).

| District | Stage 1 (yearly nodes) | Stage 2 (EAC-format flow + adj) |
|---|---|---|
| PEMS03 | [`pems03_yearly_nodes.ipynb`](xxltrafficdata/pems03_yearly_nodes.ipynb) | [`pems03_build_eac_data.ipynb`](xxltrafficdata/pems03_build_eac_data.ipynb) |
| PEMS04 | [`pems04_yearly_nodes.ipynb`](xxltrafficdata/pems04_yearly_nodes.ipynb) | [`pems04_build_eac_data.ipynb`](xxltrafficdata/pems04_build_eac_data.ipynb) |
| PEMS05 | [`pems05_yearly_nodes.ipynb`](xxltrafficdata/pems05_yearly_nodes.ipynb) | [`pems05_build_eac_data.ipynb`](xxltrafficdata/pems05_build_eac_data.ipynb) |
| PEMS06 | [`pems06_yearly_nodes.ipynb`](xxltrafficdata/pems06_yearly_nodes.ipynb) | [`pems06_build_eac_data.ipynb`](xxltrafficdata/pems06_build_eac_data.ipynb) |
| PEMS07 | [`pems07_yearly_nodes.ipynb`](xxltrafficdata/pems07_yearly_nodes.ipynb) | [`pems07_build_eac_data.ipynb`](xxltrafficdata/pems07_build_eac_data.ipynb) |
| PEMS08 | [`pems08_yearly_nodes.ipynb`](xxltrafficdata/pems08_yearly_nodes.ipynb) | [`pems08_build_eac_data.ipynb`](xxltrafficdata/pems08_build_eac_data.ipynb) |
| PEMS10 | [`pems10_yearly_nodes.ipynb`](xxltrafficdata/pems10_yearly_nodes.ipynb) | [`pems10_build_eac_data.ipynb`](xxltrafficdata/pems10_build_eac_data.ipynb) |
| PEMS11 | [`pems11_yearly_nodes.ipynb`](xxltrafficdata/pems11_yearly_nodes.ipynb) | [`pems11_build_eac_data.ipynb`](xxltrafficdata/pems11_build_eac_data.ipynb) |
| PEMS12 | [`pems12_yearly_nodes.ipynb`](xxltrafficdata/pems12_yearly_nodes.ipynb) | [`pems12_build_eac_data.ipynb`](xxltrafficdata/pems12_build_eac_data.ipynb) |

### Auxiliary notebooks / scripts

* [`pems03_build_adj.ipynb`](xxltrafficdata/pems03_build_adj.ipynb) — reference notebook for constructing the per-year adjacency matrix (used as a template for the other districts).
* [`pems03_data_quality_check.py`](xxltrafficdata/pems03_data_quality_check.py) — sanity checks for missing / duplicate stations and per-year coverage.
* [`pems_all_growth_viz.ipynb`](xxltrafficdata/pems_all_growth_viz.ipynb) — produces the cross-district growth visualisation ([`pems_all_growth_viz.pdf`](xxltrafficdata/pems_all_growth_viz.pdf), [`pems_all_growth_viz.png`](xxltrafficdata/pems_all_growth_viz.png)).
* [`pems11_adj_evolution.png`](xxltrafficdata/pems11_adj_evolution.png) — example visualisation of adjacency evolution across years for PEMS11.

<p align="center">
    <img src="xxltrafficdata/pems_all_growth_viz.png" alt="Cross-district sensor growth visualisation" align="center" width="900px" />
</p>

### Downloading the processed data

Raw PEMS dumps and the intermediate / EAC-format outputs are **not** committed to this repository. The per-district directories `pems03/ ... pems12/` and `preprocessed/` are empty placeholders.

> ☁️ **Cloud-disk download link (TODO: insert after upload):** `<paste-link-here>`

After downloading, place the files so the layout matches:

```
xxltrafficdata/
├── pems03/<year>.npz           ← yearly flow tensors
├── pems03/<year>_adj.npz       ← yearly adjacency matrices
├── pems04/...
└── preprocessed/               ← intermediate stage-1 outputs
```

Then either rerun the stage-2 notebooks to regenerate the EAC inputs, or symlink / copy the processed `<year>.npz` files into [`eac/data/`](eac/) following [`eac/README.md`](eac/README.md).

---

## 🧩 Baselines

All baselines share the same data loader ([`eac/main.py`](eac/main.py), [`eac/src/dataer/SpatioTemporalDataset.py`](eac/src/dataer/SpatioTemporalDataset.py)) and the same 3 / 6 / 12-step evaluation protocol (`eac/src/trainer/default_trainer.py::test_model`). Each method is selected through a JSON config under [`eac/conf/<DATASET>/`](eac/conf); per-dataset launch scripts live in [`eac/scripts/`](eac/scripts) (e.g. [`pems05_run.sh`](eac/scripts/pems05_run.sh), [`baselines_pems_run.sh`](eac/scripts/baselines_pems_run.sh)).

### (i) Static STGNN backbones

Trained from scratch on the current period only — used both standalone and as the shared backbone for the continual schemes.

| Baseline | Model class | Config (PEMS05 example) |
|---|---|---|
| **DCRNN** | [`DCRNN_Model`](eac/src/model/model.py) | [`conf/PEMS05/retrain_dcrnn_pems05.json`](eac/conf/PEMS05/retrain_dcrnn_pems05.json) |
| **ASTGNN** | [`ASTGNN_Model`](eac/src/model/model.py) | [`conf/PEMS05/retrain_astgnn_pems05.json`](eac/conf/PEMS05/retrain_astgnn_pems05.json) |
| **TGCN** | [`TGCN_Model`](eac/src/model/model.py) | [`conf/PEMS05/retrain_tgcn_pems05.json`](eac/conf/PEMS05/retrain_tgcn_pems05.json) |

### (ii) Naïve training schemes

Fix the backbone (`STGNN_Model`) and only vary how each period's data is used; isolate the effect of the continual strategy.

| Baseline | `strategy` | Config (PEMS05) |
|---|---|---|
| **Pretrain** | `pretrain` (train on Period 1, zero-shot afterwards) | [`pretrain_st_pems05.json`](eac/conf/PEMS05/pretrain_st_pems05.json) |
| **Retrain** | `retrain` (train from scratch each period) | [`retrain_st_pems05.json`](eac/conf/PEMS05/retrain_st_pems05.json) |
| **Online-NN** | `incremental` (fine-tune on new nodes only) | [`oneline_st_nn_pems05.json`](eac/conf/PEMS05/oneline_st_nn_pems05.json) |
| **Online-AN** | `retrain` + `load_first_year` (fine-tune all nodes from previous-period init) | [`oneline_st_an_pems05.json`](eac/conf/PEMS05/oneline_st_an_pems05.json) |

### (iii) Evolving-graph continual methods

Methods explicitly designed for streaming graphs with newly installed sensors.

| Baseline | Model class | Config (PEMS05) | Notes |
|---|---|---|---|
| **TrafficStream** (IJCAI'21) | [`TrafficStream_Model`](eac/src/model/model.py) | [`trafficstream.json`](eac/conf/PEMS05/trafficstream.json) | `incremental` + EWC + 2-hop subgraph of new nodes; drift detector in [`detect_default.py`](eac/src/model/detect_default.py) |
| **PECPM** (KDD'23) | [`PECPM_Model`](eac/src/model/model.py) | [`pecpm_pems05.json`](eac/conf/PEMS05/pecpm_pems05.json) | Pattern bank with expansion / consolidation |
| **STKEC** (TKDE'23) | [`STKEC_Model`](eac/src/model/model.py) | [`stkec.json`](eac/conf/PEMS05/stkec.json) | Influence-based node selection + learnable memory bank; trainer [`stkec_trainer.py`](eac/src/trainer/stkec_trainer.py); drift score [`detect_stkec.py`](eac/src/model/detect_stkec.py) |
| **EAC** (ICLR'25) | [`EAC_Model`](eac/src/model/model.py) | [`eac.json`](eac/conf/PEMS05/eac.json) | Frozen backbone + expand-and-compress prompt pool (`rank=6`) |

### (iv) Retrieval and test-time methods

Adapt the model without continual parameter updates on the full graph.

| Baseline | Model class | Config (PEMS05) | Notes |
|---|---|---|---|
| **STRAP** (NeurIPS'25) | [`RAP_Model`](eac/src/model/model.py) | [`strap_pems05.json`](eac/conf/PEMS05/strap_pems05.json) | Top-$K$ retrieval from a spatial/temporal/spatio-temporal pattern library |
| **ST-TTC** (NeurIPS'25) | [`STTTC_Model`](eac/src/model/model.py) | [`sttc_pems05.json`](eac/conf/PEMS05/sttc_pems05.json) | Test-time spectral calibrator + streaming FIFO memory (`use_ttc=1`); inference path `test_model_with_ttc` in [`default_trainer.py`](eac/src/trainer/default_trainer.py) |

---

## 🚀 Reproducing the main tables

| Dataset | Launch all baselines | Single-method examples |
|---|---|---|
| PEMS05 | [`scripts/pems05_run.sh`](eac/scripts/pems05_run.sh) | `python eac/main.py --conf eac/conf/PEMS05/eac.json --gpuid 0 --seed 43` |
| PEMS03–PEMS12 | [`scripts/baselines_pems_run.sh`](eac/scripts/baselines_pems_run.sh), [`scripts/extra_baselines_run.sh`](eac/scripts/extra_baselines_run.sh) | replace `PEMS05` with the target district |
| ST-TTC (separate launcher) | [`scripts/sttc_run.sh`](eac/scripts/sttc_run.sh) | — |

The aggregated numbers in [`tables/tsas_main_table_part1.tex`](tables/tsas_main_table_part1.tex) (PEMS03–PEMS07) and [`tables/tsas_main_table_part2.tex`](tables/tsas_main_table_part2.tex) (PEMS08, PEMS10–PEMS12) follow the same column order as the four baseline groups defined above.

---

## 📁 Layout

```
tsas/
├── README.md                          ← this file
├── fig/                               ← paper figures
│   ├── new1.pdf / new1.svg            ← dataset & task overview
│   └── proto.pdf                      ← method prototype
├── table/
│   └── data1.tex                      ← dataset comparison (LaTeX)
├── tables/                            ← main result tables (LaTeX)
│   ├── tsas_main_table_part1.tex      ← PEMS03–PEMS07
│   └── tsas_main_table_part2.tex      ← PEMS08, PEMS10–PEMS12
├── xxltrafficdata/                    ← data processing pipeline
│   ├── pemsXX_yearly_nodes.ipynb      ← stage 1: raw → yearly nodes
│   ├── pemsXX_build_eac_data.ipynb    ← stage 2: yearly → EAC format
│   ├── pems_all_growth_viz.ipynb      ← cross-district growth viz
│   └── pemsXX/  preprocessed/         ← empty placeholders (cloud download)
└── eac/                               ← shared codebase (all baselines)
    ├── main.py                        ← entry point
    ├── conf/PEMS{03,04,05,...,12}/    ← per-method JSON configs
    ├── src/model/                     ← model implementations
    ├── src/trainer/                   ← training / TTC loops
    └── scripts/                       ← launch scripts
```

> Heavy artefacts (raw / processed flow data, training logs) are not committed.
> Download them from the cloud-disk link in the [Dataset Processing](#-dataset-processing) section.
