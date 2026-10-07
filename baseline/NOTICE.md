# baseline/: nguồn gốc

Thư mục này chứa nguyên trạng code và dữ liệu của baseline (Setup A), lấy từ repo công khai
`kangmyoungseok/RugPull_Prediction_AI` (KITRI BoB 10th, 2021). Đây cũng là nguồn của bài
IMCOM về dự đoán rug pull trên Uniswap V2. Repo gốc **không kèm LICENSE**, nên mặc định mọi
quyền vẫn thuộc nhóm tác giả gốc.

- Khi viết báo cáo hoặc paper, cần trích dẫn bài báo và repo này làm baseline (Setup A).
- Đừng public lại thư mục này (GitHub công khai, nộp kèm source) nếu chưa hỏi tác giả.
  Code mới của dự án nằm trong `src/`, `data/sql/`, `experiments/`, `tests/` và không phụ thuộc thư mục này.

Những gì đã chỉnh so với bản gốc:
- Gỡ toàn bộ API key bị hardcode (Etherscan, Ethplorer, Bitquery). Các key này giờ đọc từ biến
  môi trường `ETHERSCAN_API_KEY`, `ETHPLORER_API_KEY`, `BITQUERY_API_KEY`.
- Xóa `__pycache__` và README gốc; đổi tên `requirements.txt` thành `requirements_original.txt`.
- Thêm `run_setup_a.py` (tái lập Setup A cộng các phép thử leakage).

Code crawl cũ (The Graph hosted service, Bitquery v1) đã ngừng hoạt động; chỉ giữ lại để tham khảo logic.
