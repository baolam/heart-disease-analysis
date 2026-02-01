# 🏥 Dự Án Phân Tích & Dự Báo Yếu Tố Bệnh Tim Mạch

Dự án này được xây dựng để phân tích sự ảnh hưởng của đa yếu tố (lối sống, di truyền, môi trường) đối với bệnh tim mạch, đồng thời sử dụng dữ liệu lịch sử để dự đoán xu hướng cho tương lai.

## 📋 Bài Toán Đặt Ra
Nghiên cứu và trả lời các câu hỏi trọng tâm:
* **Lối sống:** Các thói quen sinh hoạt và hoạt động thể chất ảnh hưởng thế nào?
* **Di truyền:** Các đặc điểm bẩm sinh chi phối tỉ lệ mắc bệnh ra sao?
* **Môi trường:** Y tế và môi trường sống đóng vai trò gì?
* **Dự báo:** Dựa trên dữ liệu lịch sử để phán đoán % chi phối của các yếu tố này trong năm tiếp theo.

---

## 📂 Cấu Trúc Dự Án
Dựa trên cấu trúc thư mục của dự án:

```text
DATA_ANALYSIS/
├── .ipynb_checkpoints/       # Lưu các bản backup của Jupyter Notebook
├── data/                     # Quản lý dữ liệu dự án
│   ├── csv/                  # Các tệp tin dữ liệu định dạng .csv
│   ├── decision/             # Dữ liệu phục vụ việc ra quyết định/mô hình (chứa các tệp file là các indicators cuối cùng cho xử lí)
│   ├── raw_concat/           # Dữ liệu thô sau khi được gộp (concatenate)
│   ├── raw_official/         # Dữ liệu thô chính thức
│   ├── raw_official_v2/      # Dữ liệu thô chính thức phiên bản 2
│   ├── raw_v1/               # Dữ liệu thô phiên bản 1
│   ├── urls/                 # Danh sách các đường dẫn nguồn dữ liệu
│   └── raw_official_v2.zip   # Bản nén của dữ liệu chính thức
├── data-analysis-env/        # Môi trường ảo (Virtual Environment) của dự án
├── docs/                     # Tài liệu hướng dẫn và ghi chú
│   └── .txt                  # Mô tả
├── preprocessing/            # Quy trình tiền xử lý dữ liệu
│   ├── .ipynb_checkpoints/
│   ├── collection/           # Module thu thập dữ liệu
│   ├── outlier/              # Xử lý các giá trị ngoại lệ (outliers)
│   ├── visualization/        # Các script/notebook phục vụ trực quan hóa
│   ├── CSV_utilization.ipynb # Notebook khai thác dữ liệu CSV
│   ├── indicator_meaning.ipynb # Giải thích ý nghĩa các chỉ số y tế
│   └── summary.ipynb         # Tổng hợp và thống kê dữ liệu
└── worldmap/                 # Chứa thông tin về bản đồ thế giới
    ├── ...