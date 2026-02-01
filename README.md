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
├── data/                  # Chứa các file dữ liệu thô và đã xử lý
├── preprocessing/         # Các script xử lý dữ liệu và handler DB
│   └── ...    # Script nạp dữ liệu vào cơ sở dữ liệu
├── worldmap/              # Hình ảnh trực quan hóa bản đồ thế giới
├── docs/                  # Tài liệu hướng dẫn và báo cáo dự án
├── Physical_activities.ipynb    # Phân tích về các hoạt động thể chất
├── collection_data_api.ipynb    # Code thu thập dữ liệu qua API
└── auto-generate-log.py         # Script tự động tạo log cho dự án