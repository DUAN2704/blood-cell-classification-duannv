# PHƯƠNG PHÁP ENSEMBLE LEARNING CẢI THIỆN BÀI TOÁN PHÂN LOẠI TẾ BÀO MÁU CÓ HÌNH THÁI TƯƠNG TỰ
Đồ án tốt nghiệp — Đại học Công nghiệp Hà Nội
Sinh viên: Nguyễn Văn Duẩn | GVHD: TS. Vũ Đình Minh

## GIỚI THIỆU:
Hệ thống phân loại tế bào máu tự động từ ảnh kính hiển vi sử dụng phương pháp Ensemble Learning kết hợp hai kiến trúc học sâu:
1. **ResNet50** — trích xuất đặc trưng cục bộ (texture, edge)  
2. **Swin Transformer** — nắm bắt quan hệ toàn cục (self-attention)
Hai mô hình được kết hợp bằng Weighted Soft Voting với cơ chế Adaptive Weighting đặc biệt cho nhóm lớp khó (MY, MMY, PMY).

## KẾT QUẢ:  
- ResNet50: **94.01% Accuracy**, F1-Macro: **0.9220**  
- Swin-S: **94.49% Accuracy**, F1-Macro: **0.9260**  
- Ensemble: **94.91% Accuracy**, F1-Macro: **0.9317**   
Cải thiện nhóm lớp khó (MY/MMY/PMY): +2.13% F1 so với ResNet50 đơn lẻ

## CẤU TRÚC:  
blood-cell-classification/  
│  
├── src/                       
│   ├── preprocess_data.py  
│   ├── train_resnet50.py  
│   ├── train_swin.py  
│   ├── ensemble.py  
│   └── ablation_study.py  
│  
├── app/                       
│   └── demo.py  
│  
│  
├── notebooks/                 
│  
├── requirements.txt  
├── .gitignore  
└── README.md  

## BỘ DỮ LIỆU:  
Gồm 2 nguồn trên kaggle:  
1. OI-PBC (PBC-YOLO)
2. Myeloblast Dataset
Dataset: 18.024 ảnh, 13 lớp tế bào máu

## CÀI ĐẶT:
git clone https://github.com/DUAN2704/blood-cell-classification-duannv.git
cd blood-cell-classification  
pip install torch torchvision timm albumentations gradio scikit-learn opencv-python pandas matplotlib seaborn

## YÊU CẦU:
Python 3.10+  
PyTorch 2.0+  
CUDA 11.8+ (khuyến nghị, có thể chạy CPU)

Tác Giả  
Nguyễn Văn Duẩn  
Mã sinh viên: 2022604282 | Lớp: 2022DHKHMT02  
Đại học Công nghiệp Hà Nội  
GVHD: TS. Vũ Đình Minh  

Giấy Phép  
Dự án này được phát triển cho mục đích học thuật.  
© 2026 Nguyễn Văn Duẩn — Đại học Công nghiệp Hà Nội