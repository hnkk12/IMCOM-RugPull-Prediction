# Đặc tả dự án: Phát hiện sớm rug pull trên Uniswap V2

> Cập nhật: 08/10/2026. File này mô tả toàn bộ dự án: mục tiêu, dữ liệu, phương pháp, cách chạy và kết quả hiện có.

---

## 1. Tổng quan

| Mục | Nội dung |
|---|---|
| **Bài toán** | Với một token mới vừa có pool thanh khoản WETH trên Uniswap V2, chỉ dùng dữ liệu on-chain trong **N giờ đầu** (N ∈ {1, 6, 24}), dự đoán pool có bị **rug pull trong 30 ngày tiếp theo** hay không. |
| **Lĩnh vực** | Data mining, pattern recognition, blockchain security |
| **Dạng bài toán** | Phân loại nhị phân trên dữ liệu bảng, theo thời gian, lệch lớp |
| **Đóng góp chính** | (1) Chứng minh baseline cũ (feature tính trên toàn vòng đời + random split) bị **leakage** nên kết quả lạc quan. (2) Đề xuất protocol phát hiện sớm không leakage: cửa sổ cố định, nhãn có censoring, temporal split có purge. (3) Tự xây dataset 23.400 pool từ dữ liệu on-chain thô. (4) Đo được cái giá của việc phát hiện sớm: rất nhiều vụ rug xảy ra trước t₀+N. |
| **Baseline** | Software artifact công khai `kangmyoungseok/RugPull-Prediction-AI` (KITRI BoB 2021). Chưa xác minh được paper/DOI đi kèm và không có bằng chứng về liên hệ với IMCOM; chỉ trích repository cho Setup A. Xem `baseline/NOTICE.md`. |

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
├─ tests/                   23 test (leakage, censoring, split, bytecode, accounting, audit) + dữ liệu giả lập
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
| pos | 8.659 | 6.080 | 3.916 |
| neg | 5.444 | 5.314 | 4.975 |
| **early_rug** | **7.953** | **10.660** | **13.161** |
| no_liquidity | 803 | 803 | 803 |
| dust | 525 | 525 | 525 |
| censored | 16 | 18 | 20 |
| **Dataset** | **14.103** (61,4% rug) | **11.394** (53,4% rug) | **8.891** (44,0% rug) |

Các số trên đã áp dụng quy tắc censoring đúng cho event dương: nếu rug đã được quan sát trong target
window thì mẫu là `pos` ngay cả khi chưa theo dõi đủ toàn bộ follow-up cần thiết để kết luận một mẫu âm.

- Có 18.161/22.597 pool có thanh khoản bị rug trong 45 ngày (**80,4%**). Trong đó 16.374 vụ qua Burn (rút LP) và 1.787 vụ qua Swap (dump token). So sánh ngưỡng dùng tolerance `1e-12` để các trường hợp giảm đúng 99% không bị loại do sai số float.
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

### 5.3 Nhóm `contract` (Setup D): 16 feature tĩnh
- 12 cờ hàm nguy hiểm: `fn_mint`, `fn_blacklist`, `fn_set_bots`, `fn_set_fee`, `fn_set_max_tx`, `fn_pause`, `fn_trading_switch`, `fn_exclude_from_fee`, `fn_cooldown`, `fn_owner`, `fn_renounce`, `fn_transfer_ownership`. Selector được tính bằng keccak từ chữ ký hàm.
- `n_functions`, `bytecode_size`, `bytecode_dup_count` (số lần template bytecode này đã được deploy trước đó)
  và `bytecode_missing` (không lấy được code hợp lệ tại thời điểm dự đoán).
- **Hạn chế:** selector được lấy xấp xỉ từ các opcode PUSH4 trong bytecode, chưa phải phân tích source.

### 5.4 Kiểm soát token accounting

Tám feature holder-share phụ thuộc vào việc tái dựng đầy đủ ledger ERC-20 từ event `Transfer`. Pipeline hiện kiểm tra
balance âm và đối chiếu tổng balance sống với circulating supply; nếu sai, tám feature này được đặt `NaN` thay vì
giữ các tỷ lệ vô lý. Có 4.195/14.103 (29,7%), 3.699/11.394 (32,5%) và 3.300/8.891 (37,1%) pool không qua kiểm tra
ở N=1/6/24h. Vì mức thiếu không ngẫu nhiên có thể tự trở thành tín hiệu, **protocol chính dùng nhóm `robust` gồm 47
feature và loại toàn bộ tám feature phụ thuộc token accounting**. Các setup đầy đủ chỉ còn là phân tích phụ.

---

## 6. Thiết kế thí nghiệm

### 6.1 Split (`src/split.py`)
- **Random:** stratified 70/15/15.
- **Temporal:** train có t₀ < 2022-01-01, val trong 2022-01-01 → 2022-07-01, test có t₀ ≥ 2022-07-01.
  **Purge** bỏ các mẫu train/val có horizon nhãn (t₀+N+H) vượt sang giai đoạn sau.
- **Cold-start:** trên test, bỏ các pool có creator đã xuất hiện trong train.
- **Template cold-start:** bỏ các pool test có bytecode template đã xuất hiện trong train.
- **Common cohort:** giao của ba dataset N=1/6/24, dùng cùng pool để tách giá trị của thời gian quan sát
  khỏi survivor-selection bias.
- **Rolling temporal:** bốn fold test liên tiếp 2022H1, 2022H2, 2023H1 và 2023H2.

### 6.2 Các setup

| Setup | Dữ liệu và feature | Split | Mục đích |
|---|---|---|---|
| **A** | Dataset_v1.9 của repo gốc, 18 feature toàn vòng đời | random 5-fold (+ temporal) | tái lập kết quả trước và thử leakage |
| **B_no_token_accounting** | cửa sổ cố định, 47 feature `robust` | random | cho thấy random split vẫn lạc quan |
| **C_no_token_accounting** | cửa sổ cố định, 47 feature `robust` | temporal + purge | **protocol chính / hiệu năng thực tế** |
| **D_no_token_accounting** | C sạch + `contract` | temporal + purge | giá trị của feature contract |
| C_repo_only | 16 feature `repo_robust` | temporal + purge | ablation: giá trị của feature mới |
| D_coldstart / D_template_coldstart | D sạch | temporal, creator/template mới | kiểm tra học thuộc; kết quả phụ |
| C_common | C sạch, cùng pool ở mọi N | temporal + purge | so sánh N không bị đổi cohort |
| C_rolling | C sạch | 4 temporal fold + purge | độ bền theo thời gian |

### 6.3 Model và metric
- **Model:** LightGBM, XGBoost (early stopping theo PR-AUC trên val), Logistic Regression (impute, signed-log, chuẩn hóa, class_weight balanced).
- **Metric:** ROC-AUC, PR-AUC, F1/precision/recall/specificity/balanced accuracy ở ngưỡng tối ưu F1
  trên **val** (dùng nguyên cho test), Brier score, ECE, confusion counts và stratified-bootstrap 95% CI.
- **Giải thích:** SHAP (TreeExplainer) cho LightGBM của Setup C sạch; các full-feature setup cũ chỉ để chẩn đoán.
- **Reproducibility:** lưu model, prediction, calibration curve, config, SHA-256 manifest và phiên bản môi trường.

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
| B sạch: random | 0.919 (0.950) | 0.911 (0.928) | 0.878 (0.858) |
| **C sạch: temporal** | **0.848 (0.860)** | **0.830 (0.796)** | **0.791 (0.653)** |
| C_repo_only | 0.821 (0.823) | 0.790 (0.735) | 0.710 (0.567) |
| D sạch: C + contract | 0.858 (0.868) | 0.834 (0.797) | 0.792 (0.659) |
| D creator cold-start | 0.858 (0.868) | 0.834 (0.797) | 0.792 (0.660) |
| D template cold-start | 0.858 (0.868) | 0.834 (0.797) | 0.792 (0.659) |

**Tỉ lệ rug trong tập test** (cần báo cáo kèm PR-AUC): B 61,4% / 53,3% / 44,1%. C/D 50,7% / 40,8% / 35,1%.

**F1 ở Setup C sạch (LightGBM):** 0.779 / 0.708 / 0.639. Kết quả chi tiết, CI, confusion counts,
Brier và ECE nằm ở `results/summary.csv`.

### 7.3 Robustness mới

**Leakage probe trên cùng dataset mới.** Chỉ dùng độ dài cửa sổ phụ thuộc nhãn đã đạt ROC-AUC
0,999/0,996/0,998 với random split và 0,972/0,992/0,986 với temporal split cho N=1/6/24h.
Điều này cô lập leakage tốt hơn việc so trực tiếp Setup A và B trên hai dataset khác nhau.

**Common cohort (8.891 pool có mặt ở cả ba N).** PR-AUC lần lượt là 0,531, 0,599 và 0,653.
Do đó, trên cùng population, quan sát thêm thực sự có ích; hiệu năng giảm trong landmark C sạch chủ yếu phản ánh
việc các rug dễ đã rời risk set trước các landmark muộn hơn.

**Operational coverage trên temporal test:** tỷ lệ tối đa còn có thể bắt tại landmark là 37,5%/24,3%/17,3%;
LightGBM thực tế bắt được 30,6%/16,7%/12,8% tổng số rug xảy ra trước hoặc trong target horizon.

**Label sensitivity:** ngưỡng 95% thêm 398 rug so với 99%; ngưỡng 97,5% thêm 251;
cumulative-99% thêm 371. Khi retrain LightGBM trên 47 feature sạch, single-95% đạt ROC-AUC 0,835/0,804/0,767
và cumulative-99% đạt 0,847/0,822/0,771 tại N=1/6/24h, cho thấy kết luận định lượng nhạy với
định nghĩa rug dù pattern temporal chính vẫn còn. Kết quả chi tiết nằm ở `results/label_sensitivity*.csv`
và các setup `C_label95`, `C_cumulative99`.

### 7.4 SHAP (Setup C sạch, LightGBM): các feature quan trọng nhất
- **N = 1h:** `n_sell`, `token_age_h`, `init_weth`, `max_drawdown`, `lp_creator_share`, `lp_top1_share`, `buy_sell_rate`, `lp_hhi`
- **N = 6h:** `lp_top1_share`, `buy_sell_rate`, `max_drawdown`, `n_sell`, `token_age_h`, `init_weth`, `unique_token_receivers`, `lp_hhi`
- **N = 24h:** `lp_top1_share` (vượt trội), `buy_sell_rate`, `swap_mean_offset`, `end_weth`, `unique_token_receivers`, `n_token_transfers`, `reserve_change`

### 7.5 Kết luận chính
1. **Baseline bị lạc quan do cửa sổ phụ thuộc nhãn.** Không diễn giải chênh lệch A/B như một hiệu ứng nhân quả
   thuần túy vì hai setup dùng dataset khác nhau; leakage probe trên cùng dataset mới là bằng chứng cô lập chính.
2. **Feature mới sạch có ích:** C cao hơn C_repo_only, ROC-AUC tăng khoảng 0,02–0,04 và PR-AUC tăng 0,04–0,07.
3. **Feature contract chỉ giúp nhỏ và không nhất quán:** tốt hơn ở 1h/6h nhưng ROC-AUC thấp hơn ở 24h. Có thể do cách lấy selector xấp xỉ.
4. **Đánh đổi của phát hiện sớm:** common-cohort cho thấy thêm thời gian có ích, nhưng operational coverage giảm mạnh
   vì nhiều vụ rug đã xảy ra trước landmark (early_rug tăng từ 7.953 lên 13.161).
5. **Độ tập trung LP** (`lp_top1_share`, `lp_hhi`) là tín hiệu mạnh và ổn định nhất qua mọi N.
6. **Kẻ scam gần như luôn dùng ví deployer mới:** cold-start gần như trùng với D (chỉ 6 pool test có creator lặp lại), nên feature lịch sử creator ít tác dụng.
7. **Không dùng missingness của ledger lỗi làm tín hiệu chính:** tám feature token holder bị loại khỏi protocol báo cáo chính.

---

## 8. Hạn chế và rủi ro
- **Mẫu có điều kiện**, không phải toàn bộ: 600 pool/tháng trong tổng số khoảng 275k cặp WETH và chỉ xét
  token deploy ≤ 7 ngày trước khi tạo pool. Tỷ lệ rug 80,4% không phải prevalence của toàn Uniswap.
- **Nhãn theo luật:** chưa bắt được honeypot (không bán được nhưng LP vẫn còn), rug chia nhỏ nhiều giao dịch (trừ khi bật `cumulative`), hay soft rug (dev bán dần).
- **Tỉ lệ rug thay đổi theo thời gian:** train 51–72% tùy landmark, test 35–51%. Ngưỡng chọn trên val có thể lệch khi đem ra test.
- **Danh sách locker còn ít**, nên `lp_locked_share` có thể bị đánh giá thấp.
- **Selector lấy từ PUSH4** là cách xấp xỉ. Crawler mới ưu tiên `eth_getCode` tại block tạo pool và ghi cờ nếu
  phải fallback sang `latest`; dữ liệu contract hiện có được crawl trước thay đổi này nên Setup D vẫn là kết quả phụ.
  `token_deployed_ts` được ước từ chênh lệch block × 12 giây.
- **Balance từ Transfer** có thể không phản ánh hoàn toàn rebasing/reflection/non-standard token. Pipeline đánh dấu ledger lỗi,
  nhưng tỷ lệ lỗi 29,7–37,1% nên tám holder-share tương ứng bị loại khỏi protocol chính.
- Nhãn 0 nghĩa là "không rug trong target horizon", không khẳng định token là benign vĩnh viễn.
- Thời điểm deploy và creator lấy từ `getcontractcreation`. Token deploy qua factory thì creator là địa chỉ gửi giao dịch.

---

## 9. Cách chạy

```powershell
# cài đặt
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest                      # 23 test

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
.venv\Scripts\python -m src.train --config experiments/setup_B_no_token_accounting.yaml
.venv\Scripts\python -m src.train --config experiments/setup_C_no_token_accounting.yaml
.venv\Scripts\python -m src.train --config experiments/setup_D_no_token_accounting.yaml
.venv\Scripts\python -m src.train --config experiments/setup_C_repo_only.yaml
.venv\Scripts\python -m src.train --config experiments/setup_D_coldstart.yaml
.venv\Scripts\python -m src.train --config experiments/setup_D_template_coldstart.yaml
.venv\Scripts\python -m src.train --config experiments/setup_C_common.yaml
.venv\Scripts\python -m src.train --config experiments/setup_C_rolling.yaml
.venv\Scripts\python -m src.build_dataset --config experiments/setup_C_label95.yaml --jobs 8
.venv\Scripts\python -m src.train --config experiments/setup_C_label95.yaml
.venv\Scripts\python -m src.build_dataset --config experiments/setup_C_cumulative99.yaml --jobs 8
.venv\Scripts\python -m src.train --config experiments/setup_C_cumulative99.yaml

# audit/robustness/reproducibility
.venv\Scripts\python -m src.leakage_probe
.venv\Scripts\python -m src.operational_eval
.venv\Scripts\python -m src.label_sensitivity
.venv\Scripts\python -m src.sample_label_audit
.venv\Scripts\python -m src.data_quality
.venv\Scripts\python -m src.paper_artifacts
.venv\Scripts\python -m src.reproducibility
```

Chạy thử không cần crawl: `python -m tests.synthetic`, rồi thêm `--data-root data/synthetic` vào `build_dataset` và `train`.

**Chia sẻ cho người khác:** code đưa lên GitHub (private). `.gitignore` đã chặn `.env`, `data/` và `baseline/`. Dữ liệu gửi qua Drive: chỉ cần `data/processed/` (đủ để train), hoặc `data/raw/` (khoảng 1.5 GB) nếu muốn build lại.

---

## 10. Việc tiếp theo

| Việc | Ưu tiên | Ghi chú |
|---|---|---|
| Hai người review hai phiếu blind trong `results/label_audit_reviewer_*.csv` | **bắt buộc trước submission** | 300 pool đã lấy mẫu phân tầng; hướng dẫn ở `paper/LABEL_AUDIT_GUIDE.md` |
| Chạy `python -m src.label_audit_report` sau adjudication | **bắt buộc trước submission** | báo cáo agreement, Cohen's κ và false-positive của rule nhãn |
| Literature matrix và citation baseline | **đã chuẩn bị** | `paper/LITERATURE_MATRIX.md`; baseline chỉ được trích như software artifact |
| Bổ sung LP locker, kiểm chứng địa chỉ | trung bình | ảnh hưởng `lp_locked_share` |
| Đối chiếu TokenSniffer/nguồn độc lập | trung bình | bổ sung cho manual label audit nếu API cho phép |
| GNN trên transaction graph (Setup E) | tùy chọn | chỉ làm nếu có câu hỏi nghiên cứu khác biệt so với temporal-GNN hiện có |
| Mở rộng sang BSC/PancakeSwap | thấp | cần nguồn dữ liệu khác |
| Viết paper | đang sẵn sàng | có thể viết Method/Dataset; khóa Results sau khi hoàn tất manual label audit |
