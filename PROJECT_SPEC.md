# Đặc tả dự án: Phát hiện sớm rug pull trên Uniswap V2

> Cập nhật: 07/10/2026. File này mô tả toàn bộ dự án: mục tiêu, dữ liệu, phương pháp, cách chạy và kết quả hiện có.

---

## 1. Tổng quan

| Mục | Nội dung |
|---|---|
| **Bài toán** | Với một token mới vừa có pool thanh khoản WETH trên Uniswap V2, chỉ dùng dữ liệu on-chain trong **N giờ đầu** (N ∈ {1, 6, 24}), dự đoán pool có bị **rug pull trong 30 ngày tiếp theo** hay không. |
| **Lĩnh vực** | Data mining, pattern recognition, blockchain security |
| **Dạng bài toán** | Phân loại nhị phân trên dữ liệu bảng, theo thời gian, lệch lớp |
| **Đóng góp chính** | (1) Chứng minh baseline cũ (feature tính trên toàn vòng đời + random split) bị **leakage** nên cho kết quả ảo. (2) Đề xuất protocol phát hiện sớm không leakage: cửa sổ cố định, nhãn có censoring, temporal split có purge. (3) Tự xây dataset 23.400 pool từ dữ liệu on-chain thô. (4) Đo được cái giá của việc phát hiện sớm: rất nhiều vụ rug xảy ra trước t₀+N. |
| **Baseline** | Repo công khai `kangmyoungseok/RugPull_Prediction_AI` (KITRI BoB 2021, bài IMCOM). Phải trích dẫn khi viết paper. Xem `baseline/NOTICE.md`. |

---

## 2. Cấu trúc thư mục

```
IMCOM-RugPull-Prediction/
├─ PROJECT_SPEC.md          file này
├─ README.md                hướng dẫn chạy ngắn gọn
├─ requirements.txt         thư viện (Python 3.12)
├─ .env                     ETHERSCAN_API_KEY (KHÔNG commit)
├─ .env.example
├─ baseline/                repo gốc, giữ để tái lập Setup A (không public lại)
│  ├─ NOTICE.md             nguồn gốc + những gì đã chỉnh
│  ├─ run_setup_a.py        Setup A + các phép thử leakage
│  └─ Dataset_v1.9.csv, ann97.h5, 1. Labeling/, 2. Feature/ ...
├─ data/
│  ├─ sql/                  query BigQuery (phương án trả phí, chưa dùng)
│  ├─ raw/                  dữ liệu crawl (~1.5 GB)
│  │  ├─ pairs.parquet      23.400 pool được lấy mẫu
│  │  ├─ contracts.parquet  bytecode/selector của 23.400 token
│  │  ├─ events/YYYY-MM.parquet   24,9 triệu event
│  │  └─ etherscan/         cache crawl (pair_created.jsonl, creation.json, pools/)
│  └─ processed/
│     ├─ rugs.parquet       kết quả phát hiện rug từng pool
│     ├─ dataset_N{1,6,24}h.parquet   dataset để train
│     └─ label_status.csv   thống kê nhãn
├─ src/
│  ├─ constants.py          địa chỉ, topic event, locker, chữ ký hàm nguy hiểm
│  ├─ config.py             đọc YAML (có `extends`)
│  ├─ etherscan_fetch.py    crawler miễn phí (đang dùng)
│  ├─ bq_fetch.py           crawler BigQuery (cần billing)
│  ├─ decode.py             event thô -> bảng pool/LP/token/owner
│  ├─ label.py              luật rug pull + censoring
│  ├─ features.py           feature trong cửa sổ [t₀, t₀+N]
│  ├─ contract_features.py  feature tĩnh từ bytecode (Setup D)
│  ├─ split.py              random / temporal split (có purge)
│  ├─ build_dataset.py      raw -> processed
│  └─ train.py              train + đánh giá + SHAP
├─ experiments/             common.yaml, setup_B/C/D.yaml, ablation
├─ tests/                   16 test (gồm test chống leakage) + dữ liệu giả lập
└─ results/                 metrics, dự đoán, SHAP, summary.csv
```

---

## 3. Dữ liệu

### 3.1 Nguồn
**Etherscan API V2** (gói miễn phí: 5 request/giây, 100k request/ngày). Riêng bytecode lấy qua RPC công khai `ethereum-rpc.publicnode.com`.

Ban đầu dự định dùng **BigQuery `crypto_ethereum`**, nhưng một lần quét bảng `logs` cho cả giai đoạn tốn khoảng 1.2 TB, mà pipeline cần 2 lần quét (khoảng 2.4 TB), vượt free tier 1 TB/tháng. Code BigQuery vẫn giữ trong `src/bq_fetch.py` và `data/sql/`. Có thể dùng khi có billing.

### 3.2 Quy trình crawl (`src/etherscan_fetch.py`)

| Bước | Làm gì | Số request |
|---|---|---|
| `pairs` | Lấy mọi event `PairCreated` từ factory Uniswap V2 có WETH, giai đoạn 2020-10-01 → 2024-01-01 | 367 → **275.022 cặp WETH** |
| `sample` | Lấy ngẫu nhiên khoảng 960 ứng viên/tháng, tra ai deploy token (`getcontractcreation`), chỉ giữ token **deploy ≤ 7 ngày** trước khi tạo pool, rồi lấy **600 pool/tháng** | 7.472 → **23.400 pool** (39 tháng × 600) |
| `pools` | Với mỗi pool: lấy log của địa chỉ pair (Sync/Mint/Burn/Swap trong 45 ngày, Transfer LP trong 3 ngày) và log của token (Transfer, OwnershipTransferred từ lúc deploy đến 3 ngày sau khi tạo pool), cộng bytecode | khoảng 108.700 |
| `export` | Gộp cache thành parquet theo tháng | 0 |

Crawler có cache nên chạy lại vẫn tiếp tục được, chạy 16 luồng song song nhưng dùng chung giới hạn tốc độ.

### 3.3 Hằng số on-chain
- Factory Uniswap V2: `0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f`
- WETH: `0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2`
- Topic event đã kiểm chứng bằng keccak (PairCreated, Sync, Mint, Burn, Swap, Transfer, OwnershipTransferred).
- LP locker: Unicrypt, Team.Finance. **Cần kiểm tra lại và bổ sung.**

---

## 4. Gán nhãn (`src/label.py`)

### 4.1 Phát hiện rug
- **t₀** là lúc Mint đầu tiên (pool có thanh khoản), không phải lúc PairCreated.
- WETH reserve lấy từ event **Sync**. Uniswap V2 phát Sync ngay trước Mint/Burn/Swap trong cùng giao dịch, nên event ở `log_index + 1` cho biết hành động nào gây ra thay đổi.
- **Rug** là lần Burn hoặc Swap đầu tiên làm WETH reserve **giảm ≥ 99% trong một event**. Giữ luật này từ repo gốc.
- Với rug qua **Swap**, chỉ tính khi lượng token swap vào **> 5 lần thanh khoản token ban đầu**. Đây là bộ lọc của repo gốc để loại báo động giả do MEV/sandwich.
- Tùy chọn ablation `cumulative`: reserve xuống dưới 1% mức cao nhất trong vòng 1 giờ (bắt rug chia nhỏ nhiều giao dịch). Mặc định tắt.

### 4.2 Nhãn cho phát hiện sớm (H = 30 ngày)

| Trạng thái | Điều kiện | Dùng để train? |
|---|---|---|
| `pos` | rug trong (t₀+N, t₀+N+H] | có, nhãn 1 |
| `neg` | không rug đến t₀+N+H, đã quan sát đủ | có, nhãn 0 |
| `early_rug` | rug ≤ t₀+N (quá muộn để dự đoán) | không |
| `censored` | dữ liệu chưa phủ hết t₀+N+H | không |
| `dust` | thanh khoản ban đầu < 0.1 ETH | không |
| `no_liquidity` | pool không bao giờ có Mint | không |

### 4.3 Thống kê nhãn thực tế

| Trạng thái | N = 1h | N = 6h | N = 24h |
|---|---|---|---|
| pos | 8.246 | 5.645 | 3.389 |
| neg | 5.853 | 5.745 | 5.511 |
| **early_rug** | **7.946** | **10.653** | **13.145** |
| no_liquidity | 803 | 803 | 803 |
| dust | 525 | 525 | 525 |
| censored | 27 | 29 | 27 |
| **Dataset** | **14.099** (58,5% rug) | **11.390** (49,6% rug) | **8.900** (38,1% rug) |

- Có 17.304/22.597 pool có thanh khoản bị rug trong 45 ngày (**76,6%**). Trong đó 15.517 vụ qua Burn (rút LP) và 1.787 vụ qua Swap (dump token).
- Khoảng **35% số pool có thanh khoản đã rug ngay trong giờ đầu**, và khoảng 58% trong 24 giờ đầu. Đây là giới hạn cố hữu của phát hiện sớm.

---

## 5. Feature (`src/features.py`, `src/contract_features.py`)

**Nguyên tắc chống leakage:** mọi feature chỉ dùng event có `ts ≤ t₀ + N`. Có test kiểm chứng: xóa hết event sau t₀+N thì feature không đổi (`tests/test_features_ignore_the_future`).

### 5.1 Nhóm `repo`: 18 feature của baseline, viết lại theo cửa sổ

| Baseline (toàn vòng đời) | Bản trong cửa sổ |
|---|---|
| mint/burn_count_per_week | `n_mint`, `n_burn` (đếm trong cửa sổ) |
| mint/swap/burn_ratio | `mint_ratio`, `swap_ratio`, `burn_ratio` |
| mint/swap/burn_mean_period | `*_mean_offset` = trung bình (ts − t₀)/N |
| swap_in/out_per_week, swap_rate | `n_buy`, `n_sell`, `buy_sell_rate` |
| lp_avg, lp_std | `lp_avg`, `lp_std` (theo phần trăm nắm giữ LP) |
| lp_creator_holding_ratio | `lp_creator_share` |
| lp_lock_ratio | `lp_locked_share` (LP nằm ở locker) |
| token_burn_ratio | `token_burned_share` |
| token_creator_holding_ratio | `creator_token_share` |
| number_of_token_creation_of_creator | `creator_prev_tokens` (**chỉ đếm token tạo trước**; baseline đếm cả token tạo sau, cũng là một dạng leakage) |

### 5.2 Nhóm `new`: 37 feature mới
- **Dòng tiền và hoạt động:** `n_swap`, `events_per_hour`, `buy_vol_eth`, `sell_vol_eth`, `net_flow_eth`, `sell_buy_vol_ratio`, `unique_buyers`
- **Sniping lúc launch:** `first_swap_delay`, `swaps_in_launch_block`, `frac_swaps_first_5min`
- **Thanh khoản và giá:** `init_weth`, `end_weth`, `max_weth`, `reserve_change`, `max_drawdown`, `price_change`, `price_vol`, `burn_weth_ratio`
- **Độ tập trung LP:** `n_lp_providers`, `lp_holder_count`, `lp_hhi`, `lp_top1_share`, `lp_burned_share`
- **Độ tập trung token:** `token_holder_count`, `token_hhi`, `token_top1_share`, `token_top10_share`, `pair_token_share`, `contract_self_share`, `post_launch_mint_ratio`, `n_token_transfers`, `unique_token_receivers`
- **Token và owner:** `token_age_h`, `owner_renounced`, `n_owner_changes`, `creator_is_lp_provider`, `creator_prev_rugs` (rug trước đó của cùng creator, chỉ tính rug xảy ra trước t₀+N)

### 5.3 Nhóm `contract` (Setup D): 15 feature tĩnh
- 12 cờ hàm nguy hiểm: `fn_mint`, `fn_blacklist`, `fn_set_bots`, `fn_set_fee`, `fn_set_max_tx`, `fn_pause`, `fn_trading_switch`, `fn_exclude_from_fee`, `fn_cooldown`, `fn_owner`, `fn_renounce`, `fn_transfer_ownership`. Selector được tính bằng keccak từ chữ ký hàm.
- `n_functions`, `bytecode_size`, `bytecode_dup_count` (số lần template bytecode này đã được deploy trước đó).
- **Hạn chế:** selector được lấy xấp xỉ từ các opcode PUSH4 trong bytecode, chưa phải phân tích source.

---

## 6. Thiết kế thí nghiệm

### 6.1 Split (`src/split.py`)
- **Random:** stratified 70/15/15.
- **Temporal:** train có t₀ < 2022-01-01, val trong 2022-01-01 → 2022-07-01, test có t₀ ≥ 2022-07-01.
  **Purge** bỏ các mẫu train/val có horizon nhãn (t₀+N+H) vượt sang giai đoạn sau.
- **Cold-start:** trên test, bỏ các pool có creator đã xuất hiện trong train.

### 6.2 Các setup

| Setup | Dữ liệu và feature | Split | Mục đích |
|---|---|---|---|
| **A** | Dataset_v1.9 của repo gốc, 18 feature toàn vòng đời | random 5-fold (+ temporal) | tái lập con số ảo, thử leakage |
| **B** | cửa sổ cố định, `repo` + `new` | random | cho thấy random split vẫn lạc quan |
| **C** | cửa sổ cố định, `repo` + `new` | temporal + purge | **hiệu năng thực tế** |
| **D** | C + `contract` | temporal + purge | giá trị của feature contract |
| C_repo_only | chỉ `repo` | temporal + purge | ablation: giá trị của feature mới |
| D_coldstart | D | temporal, chỉ creator mới | creator chưa từng thấy |

### 6.3 Model và metric
- **Model:** LightGBM, XGBoost (early stopping theo PR-AUC trên val), Logistic Regression (impute, signed-log, chuẩn hóa, class_weight balanced).
- **Metric:** ROC-AUC, PR-AUC, F1/precision/recall ở ngưỡng tối ưu F1 trên **val** (dùng nguyên cho test), Brier score.
- **Giải thích:** SHAP (TreeExplainer) cho LightGBM/XGBoost ở Setup C/D.

---

## 7. Kết quả

### 7.1 Setup A: repo gốc và các phép thử leakage (XGBoost, trừ khi ghi khác)

| Thí nghiệm | ROC-AUC | Ghi chú |
|---|---|---|
| 18 feature, random 5-fold | **0.990** | đúng như repo gốc báo cáo |
| 18 feature, ANN (kiến trúc của `ann97.h5`) | 0.970 | |
| **Chỉ dùng độ dài vòng đời pool** | **0.953** | một con số về thời gian đã đủ |
| **Chỉ dùng độ dài cửa sổ tính feature** | **0.958** | trung vị: rug ≈ 1 giờ, sạch ≈ 60 ngày |
| Bỏ các feature chuẩn hóa theo thời gian | 0.979 | leakage nằm ở mốc chụp feature, không chỉ ở vài cột |
| 18 feature, temporal 70/30 | 0.988 (ANN 0.951) | đổi split thôi là không đủ |

Dataset gốc có 90% mẫu là rug, nên accuracy và PR-AUC ở đây bị đội lên. Hãy so sánh bằng ROC-AUC.

### 7.2 Setup B/C/D: ROC-AUC (PR-AUC), LightGBM

| Setup | N = 1h | N = 6h | N = 24h |
|---|---|---|---|
| B: random | 0.942 (0.960) | 0.941 (0.942) | 0.924 (0.891) |
| **C: temporal** | **0.897 (0.882)** | **0.890 (0.825)** | **0.842 (0.661)** |
| C_repo_only | 0.873 (0.836) | 0.850 (0.753) | 0.794 (0.579) |
| D: C + contract | 0.904 (0.889) | 0.893 (0.835) | 0.843 (0.663) |
| D_coldstart | 0.904 (0.889) | 0.893 (0.835) | 0.843 (0.664) |

**Tỉ lệ rug trong tập test** (cần báo cáo kèm PR-AUC): B 58,5% / 49,6% / 38,1%. C/D 44,7% / 32,9% / 23,5%.

**F1 ở Setup C (LightGBM):** 0.811 / 0.771 / 0.650. Kết quả chi tiết của cả 3 model nằm ở `results/summary.csv`.

### 7.3 SHAP (Setup C, LightGBM): các feature quan trọng nhất
- **N = 1h:** `n_sell`, `lp_top1_share`, `token_age_h`, `max_drawdown`, `init_weth`, `lp_creator_share`, `lp_hhi`, `buy_sell_rate`
- **N = 6h:** `lp_top1_share`, `max_drawdown`, `n_sell`, `lp_hhi`, `init_weth`, `buy_sell_rate`, `token_age_h`
- **N = 24h:** `lp_top1_share` (vượt trội), `buy_sell_rate`, `end_weth`, `lp_hhi`, `pair_token_share`, `n_token_transfers`

### 7.4 Kết luận chính
1. **Kết quả đi theo đúng thứ tự A (0.99) > B (0.94) > C (0.84–0.90).** Con số của baseline là ảo. Leakage đến từ việc cửa sổ tính feature phụ thuộc nhãn, và chỉ đổi split thì không khắc phục được.
2. **Feature mới có ích:** C cao hơn C_repo_only, ROC-AUC tăng 0.02–0.05 và PR-AUC tăng 0.05–0.08.
3. **Feature contract chỉ giúp rất ít** (khoảng +0.01 ROC-AUC). Có thể do cách lấy selector xấp xỉ.
4. **Đánh đổi của phát hiện sớm:** N lớn thì có thêm thông tin, nhưng nhiều vụ rug đã xảy ra trước đó (early_rug tăng từ 7.946 lên 13.145) và model cũng đoán khó hơn trên phần còn lại.
5. **Độ tập trung LP** (`lp_top1_share`, `lp_hhi`) là tín hiệu mạnh và ổn định nhất qua mọi N.
6. **Kẻ scam gần như luôn dùng ví deployer mới:** cold-start gần như trùng với D (chỉ 6 pool test có creator lặp lại), nên feature lịch sử creator ít tác dụng.

---

## 8. Hạn chế và rủi ro
- **Mẫu**, không phải toàn bộ: 600 pool/tháng trong tổng số khoảng 275k cặp WETH. Chỉ xét token deploy ≤ 7 ngày trước khi tạo pool.
- **Nhãn theo luật:** chưa bắt được honeypot (không bán được nhưng LP vẫn còn), rug chia nhỏ nhiều giao dịch (trừ khi bật `cumulative`), hay soft rug (dev bán dần).
- **Tỉ lệ rug thay đổi theo thời gian:** train 71% rug, test 24–45%. Ngưỡng chọn trên val có thể lệch khi đem ra test.
- **Danh sách locker còn ít**, nên `lp_locked_share` có thể bị đánh giá thấp.
- **Selector lấy từ PUSH4** là cách xấp xỉ. `token_deployed_ts` được ước từ chênh lệch block × 12 giây.
- Thời điểm deploy và creator lấy từ `getcontractcreation`. Token deploy qua factory thì creator là địa chỉ gửi giao dịch.

---

## 9. Cách chạy

```powershell
# cài đặt
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest                      # 16 test

# Setup A
.venv\Scripts\python baseline\run_setup_a.py

# crawl (cần .env có ETHERSCAN_API_KEY; mất khoảng 1 ngày)
.venv\Scripts\python -m src.etherscan_fetch pairs
.venv\Scripts\python -m src.etherscan_fetch sample
.venv\Scripts\python -m src.etherscan_fetch pools
.venv\Scripts\python -m src.etherscan_fetch export

# gán nhãn + feature (khoảng 20 phút, cần khoảng 10 GB RAM)
.venv\Scripts\python -m src.build_dataset

# train
.venv\Scripts\python -m src.train --config experiments/setup_B.yaml
.venv\Scripts\python -m src.train --config experiments/setup_C.yaml
.venv\Scripts\python -m src.train --config experiments/setup_D.yaml
.venv\Scripts\python -m src.train --config experiments/setup_C_repo_only.yaml
.venv\Scripts\python -m src.train --config experiments/setup_D_coldstart.yaml
```

Chạy thử không cần crawl: `python -m tests.synthetic`, rồi thêm `--data-root data/synthetic` vào `build_dataset` và `train`.

**Chia sẻ cho người khác:** code đưa lên GitHub (private). `.gitignore` đã chặn `.env`, `data/` và `baseline/`. Dữ liệu gửi qua Drive: chỉ cần `data/processed/` (đủ để train), hoặc `data/raw/` (khoảng 1.5 GB) nếu muốn build lại.

---

## 10. Việc tiếp theo

| Việc | Ưu tiên | Ghi chú |
|---|---|---|
| GNN trên transaction graph (Setup E) | cao | có trong topic. Đã có sẵn Transfer/Swap trong cửa sổ để dựng graph ví ↔ token. Dùng GraphSAGE/GIN (PyTorch Geometric) |
| Đối chiếu nhãn với TokenSniffer | trung bình | lấy mẫu vài trăm token, đo mức khớp, chỉ dùng để kiểm chứng |
| Bổ sung LP locker, kiểm chứng địa chỉ | trung bình | ảnh hưởng `lp_locked_share` |
| Ablation `cumulative` và các ngưỡng 95%/99% | thấp | độ nhạy của nhãn |
| Đánh giá kèm early_rug: recall trên toàn bộ vụ rug | trung bình | cho biết phát hiện sớm bắt được bao nhiêu % tổng số vụ rug |
| Mở rộng sang BSC/PancakeSwap | thấp | cần nguồn dữ liệu khác |
| Viết paper | chờ | luận điểm chính: A ảo, C thực, cái giá của việc chờ thêm |
