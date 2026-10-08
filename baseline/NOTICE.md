# baseline/: nguồn phần mềm tham chiếu

Thư mục này lưu bản chụp code và dữ liệu của dự án công khai
[`kangmyoungseok/RugPull-Prediction-AI`](https://github.com/kangmyoungseok/RugPull-Prediction-AI),
được README của repository mô tả là dự án Team BOBAI tại KITRI Best of the Best lần thứ 10
(09–12/2021). Repository gốc **không kèm LICENSE**; vì vậy mọi quyền vẫn thuộc nhóm tác giả gốc.

Không có bằng chứng thư mục này là artifact của một bài IMCOM. Đã kiểm tra chương trình và mục lục
proceedings IMCOM 2022 nhưng không tìm thấy bài rug-pull tương ứng. Cho đến khi tác giả xác nhận một
ấn phẩm cụ thể, paper của dự án này chỉ được phép trích repository như **software artifact**; không tự
gán tiêu đề bài báo, DOI hay venue IMCOM cho nó.

- Khi viết paper, trích repository ở trên cho Setup A và ghi rõ ngày truy cập.
- Không public lại thư mục này hoặc đưa nó vào supplementary material nếu chưa có sự cho phép của tác giả.
  Code mới của dự án nằm trong `src/`, `data/sql/`, `experiments/` và `tests/`, không phụ thuộc thư mục này.

Những thay đổi cục bộ so với bản gốc:

- Gỡ API key hardcode (Etherscan, Ethplorer, Bitquery); các key được đọc từ biến môi trường.
- Xóa cache và đổi tên `requirements.txt` thành `requirements_original.txt`.
- Thêm `run_setup_a.py` để tái lập Setup A và chạy các phép thử leakage.

Code crawl cũ dùng The Graph hosted service và Bitquery v1 đã ngừng hoạt động; chỉ giữ để tham khảo logic.
