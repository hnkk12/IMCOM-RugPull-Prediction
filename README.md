# Early Rug-Pull Detection on Uniswap V2

Dự đoán một token mới trên Uniswap V2 (cặp với WETH) có bị rug pull trong **H = 30 ngày** tới hay không,
chỉ dựa trên dữ liệu on-chain của **N giờ đầu** sau khi pool có thanh khoản (N ∈ {1, 6, 24}).

Thông điệp chính của project: protocol kiểu cũ (feature tính trên toàn bộ vòng đời + random split, Setup A)
cho ra con số ảo. Setup C (cửa sổ cố định + temporal split) mới phản ánh hiệu năng thực tế của early detection.

## Cấu trúc

```
baseline/            repo gốc, giữ nguyên để tái lập Setup A (xem baseline/NOTICE.md)
  run_setup_a.py     Setup A + các phép thử leakage
data/
  sql/               01_pairs.sql, 02_events.sql, 03_contracts.sql (BigQuery crypto_ethereum)
  raw/               parquet crawl về (pairs, contracts, events/YYYY-MM.parquet)
  processed/         rugs.parquet, dataset_N{1,6,24}h.parquet, label_status.csv
src/
  bq_fetch.py        crawl BigQuery -> parquet (có dry-run chặn chi phí)
  decode.py          event thô -> pool_ev / lp_tr / tok_tr / own_ev
  label.py           luật rug pull + censoring
  features.py        extract_features(pool, t0, N): chỉ dùng event trong [t0, t0+N]
  contract_features.py  feature tĩnh từ bytecode (Setup D)
  split.py           random / temporal (có purge)
  train.py           LightGBM, XGBoost, LogReg + SHAP
  build_dataset.py   raw -> processed
experiments/         common.yaml + setup_B/C/D (+ ablation)
tests/               test đơn vị + dữ liệu giả lập (synthetic.py)
results/             metrics.csv, SHAP, summary.csv cho từng setup
```

## Cài đặt

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest            # 14 test, gồm test chống leakage
```

## Quy trình

### Bước 1: Setup A (repo gốc)
```powershell
.venv\Scripts\python baseline\run_setup_a.py
```
Script chạy XGBoost và ANN (cùng kiến trúc với `ann97.h5`) bằng 5-fold CV trên `Dataset_v1.9.csv`,
kèm các phép thử leakage:
- `lifecycle_only`: chỉ dùng độ dài vòng đời pool.
- `window_only`: chỉ dùng độ dài cửa sổ mà repo gốc dùng để tính feature.
  Với token rug, cửa sổ chạy tới lúc rug (trung vị khoảng 1 giờ); với token sạch thì trung vị khoảng 60 ngày.
- `repo18_no_time`: bỏ các feature chuẩn hóa theo thời gian.
- `repo18_temporal`: train trên pool cũ, test trên pool mới hơn.

### Bước 2a: Crawl miễn phí bằng Etherscan API (đang dùng)
Tạo file `.env` ở thư mục gốc, nội dung `ETHERSCAN_API_KEY=...` (key miễn phí lấy ở etherscan.io/myapikey), rồi chạy:
```powershell
.venv\Scripts\python -m src.etherscan_fetch pairs    # toàn bộ PairCreated có WETH (vài trăm call)
.venv\Scripts\python -m src.etherscan_fetch sample   # lấy mẫu 600 pool/tháng, chỉ giữ token mới (deploy <= 7 ngày)
.venv\Scripts\python -m src.etherscan_fetch pools    # crawl log từng pool, ~4-5 call/pool, mất 1-2 ngày
.venv\Scripts\python -m src.etherscan_fetch export
```
Gói free giới hạn 100k call/ngày. Mọi bước đều có cache nên có thể dừng giữa chừng và chạy lại lệnh cũ để làm tiếp.
Selector của Setup D được lấy từ các opcode PUSH4 trong bytecode. Đây là cách xấp xỉ, nên ghi rõ trong paper.

### Bước 2b: Crawl dữ liệu bằng BigQuery (cần billing, khoảng 2.4 TB)
```powershell
gcloud auth application-default login
$env:GOOGLE_CLOUD_PROJECT = "your-project"
.venv\Scripts\python -m src.bq_fetch pairs
.venv\Scripts\python -m src.bq_fetch events      # query lớn nhất: dry-run báo số GB trước khi chạy
.venv\Scripts\python -m src.bq_fetch contracts
.venv\Scripts\python -m src.bq_fetch export
```
`python -m src.bq_fetch sql` in ra SQL đã render để dán vào BigQuery console nếu muốn.
Mọi query đều dry-run trước. Query nào quét quá `max_gb_per_query` thì dừng, trừ khi thêm `--yes`.

### Bước 3–4: Gán nhãn + feature
```powershell
.venv\Scripts\python -m src.build_dataset --jobs 8
```

**Nhãn** (`src/label.py`, giữ luật của repo gốc nhưng tính trên reserve từ event Sync):
- Rug là lần Burn/Swap đầu tiên làm WETH trong pool giảm ≥ 99% chỉ trong một event.
  Với rug qua Swap, chỉ tính khi lượng token swap vào > 5 lần thanh khoản token ban đầu (lọc MEV như repo gốc).
- t₀ là lúc Mint đầu tiên (lúc pool có thanh khoản), không phải lúc PairCreated.
- `pos`: rug rơi vào (t₀+N, t₀+N+H]. `neg`: không rug và đã quan sát đủ N+H.
- Bị loại: `early_rug` (rug trước t₀+N), `censored` (chưa đủ N+H dữ liệu), `dust` (thanh khoản ban đầu < 0.1 ETH).
- File `label_status.csv` cho biết bao nhiêu % rug xảy ra trước t₀+N. Đây chính là giới hạn của early detection.

**Feature** (`src/features.py`):
- `repo`: 18 feature của repo gốc, viết lại theo cửa sổ (đếm, tỉ lệ, offset/N).
  Không còn "per week" hay "mean period" tính trên toàn vòng đời.
- `new`: HHI và top-share của holder token và holder LP, phần LP bị lock/burn, phần token creator đang giữ,
  dòng tiền mua/bán, drawdown của reserve, sniping trong block launch, renounce ownership,
  tuổi token tính đến lúc tạo pool, số token và số rug trước đó của cùng creator (chỉ nhìn về quá khứ).
- `contract` (Setup D): cờ hàm nguy hiểm (mint, blacklist, setBots, setFee, pause, openTrading...)
  lấy từ `function_sighashes`, kích thước bytecode, số lần template bytecode đã được dùng trước đó.

### Bước 5: Train các setup
```powershell
.venv\Scripts\python -m src.train --config experiments/setup_B.yaml   # cửa sổ cố định + random split
.venv\Scripts\python -m src.train --config experiments/setup_C.yaml   # cửa sổ cố định + temporal split (+SHAP)
.venv\Scripts\python -m src.train --config experiments/setup_D.yaml   # C + feature contract
.venv\Scripts\python -m src.train --config experiments/setup_C_repo_only.yaml   # ablation
.venv\Scripts\python -m src.train --config experiments/setup_D_coldstart.yaml   # chỉ creator chưa từng thấy
```
Kết quả gồm ROC-AUC, **PR-AUC** (metric chính, vì dữ liệu lệch lớp), F1/precision/recall ở ngưỡng chọn trên tập val,
và Brier score, ghi ra `results/setup_*/metrics.csv` và `results/summary.csv`.

### Chạy thử không cần BigQuery
```powershell
.venv\Scripts\python -m tests.synthetic
.venv\Scripts\python -m src.build_dataset --data-root data/synthetic
.venv\Scripts\python -m src.train --config experiments/setup_C.yaml --data-root data/synthetic
```
Dữ liệu giả lập cố ý tách lớp rất rõ (AUC ≈ 1). Lần chạy này chỉ để kiểm tra pipeline chạy thông, đừng báo cáo con số này.

## Ghi chú
- Split temporal mặc định: train có t₀ < 2022-01-01, val từ 2022-01 đến 2022-06, test từ 2022-07 trở đi.
  Purge sẽ bỏ các mẫu train có horizon nhãn chồng sang giai đoạn val.
- `LP_LOCKERS` trong `src/constants.py` mới có Unicrypt và Team.Finance. Nên kiểm tra lại trên Etherscan và bổ sung thêm.
- GNN trên transaction graph và Slither (cần source đã verify) là bước mở rộng sau, chưa nằm trong pipeline này.
