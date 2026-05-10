import os
import sys
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import timm

print(f"Gradio  : {gr.__version__}")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")

# CẤU HÌNH ĐƯỜNG DẪN
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESNET_PATH  = os.path.join(BASE_DIR, "resnet50_best.pth")
SWIN_PATH    = os.path.join(BASE_DIR, "swin_best.pth")
EXAMPLES_DIR = os.path.join(BASE_DIR, "examples")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(RESNET_PATH):
    print(f"Không tìm thấy: {RESNET_PATH}")
    print("Copy file resnet50_best.pth vào cùng thư mục với demo.py")
    sys.exit(1)

if not os.path.exists(SWIN_PATH):
    print(f"Không tìm thấy: {SWIN_PATH}")
    print("Copy file swin_best.pth vào cùng thư mục với demo.py")
    sys.exit(1)

print(f"Tìm thấy: {RESNET_PATH}")
print(f"Tìm thấy: {SWIN_PATH}")
print(f"Device  : {DEVICE}")

# CẤU HÌNH MÔ HÌNH
CLASSES = [
    "BA", "BNE", "EO", "ERB", "IG", "LY", "MMY",
    "MO", "MY", "MYELOBLAST", "PLATELET", "PMY", "SNE"
]
NUM_CLASSES = len(CLASSES)

CLASS_NAMES = {
    "BA"        : "Basophil — Bạch cầu ưa kiềm",
    "BNE"       : "Band Neutrophil — Bạch cầu trung tính dạng dải",
    "EO"        : "Eosinophil — Bạch cầu ưa acid",
    "ERB"       : "Erythroblast — Nguyên hồng cầu",
    "IG"        : "Immature Granulocyte — Hạt bào chưa trưởng thành",
    "LY"        : "Lymphocyte — Bạch cầu lympho",
    "MMY"       : "Metamyelocyte — Hậu tủy bào",
    "MO"        : "Monocyte — Bạch cầu đơn nhân",
    "MY"        : "Myelocyte — Tủy bào",
    "MYELOBLAST": "Myeloblast — Nguyên tủy bào",
    "PLATELET"  : "Platelet — Tiểu cầu",
    "PMY"       : "Promyelocyte — Tiền tủy bào",
    "SNE"       : "Segmented Neutrophil — Bạch cầu trung tính phân múi",
}

HARD_CLASSES  = ["MY", "MMY", "PMY"]

# Cập nhật theo kết quả grid search thực tế từ weight_search.csv
W_RESNET      = 0.7
W_SWIN        = 0.3
W_HARD_RESNET = 0.3
W_HARD_SWIN   = 0.7
T_RESNET      = 1.2
T_SWIN        = 1.0

MODEL_STATS = {
    "ResNet50": {"accuracy": 94.01, "f1": 0.9220},
    "Swin-S"  : {"accuracy": 94.49, "f1": 0.9260},
    "Ensemble": {"accuracy": 94.91, "f1": 0.9317},
}

# LOAD MÔ HÌNH
def build_resnet50(num_classes):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    return model

def build_swin(num_classes):
    backbone = timm.create_model(
        "swin_small_patch4_window7_224",
        pretrained=False, num_classes=0, global_pool="avg"
    )
    class SwinClassifier(nn.Module):
        def __init__(self, backbone, in_features, num_classes):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, 384),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(384, num_classes)
            )
        def forward(self, x):
            return self.head(self.backbone(x))
    return SwinClassifier(backbone, backbone.num_features, num_classes)

print("\nĐang load mô hình...")

resnet_model = build_resnet50(NUM_CLASSES).to(DEVICE)
swin_model   = build_swin(NUM_CLASSES).to(DEVICE)

ckpt_r = torch.load(RESNET_PATH, map_location=DEVICE)
ckpt_s = torch.load(SWIN_PATH,   map_location=DEVICE)
resnet_model.load_state_dict(ckpt_r["model_state"])
swin_model.load_state_dict(ckpt_s["model_state"])
resnet_model.eval()
swin_model.eval()

print(f"ResNet50 đã load (epoch={ckpt_r['epoch']}, val_acc={ckpt_r['val_acc']:.4f})")
print(f"Swin-S   đã load (epoch={ckpt_s['epoch']}, val_acc={ckpt_s['val_acc']:.4f})")
print("Sẵn sàng demo!\n")

# TRANSFORM VÀ INFERENCE
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict(image: Image.Image):
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob_r = torch.softmax(resnet_model(tensor) / T_RESNET, dim=1).cpu().numpy()[0]
        prob_s = torch.softmax(swin_model(tensor)   / T_SWIN,   dim=1).cpu().numpy()[0]

    prob_ens = W_RESNET * prob_r + W_SWIN * prob_s
    for cls in HARD_CLASSES:
        idx = CLASSES.index(cls)
        prob_ens[idx] = W_HARD_RESNET * prob_r[idx] + W_HARD_SWIN * prob_s[idx]

    return prob_r, prob_s, prob_ens

# HÀM XỬ LÝ OUTPUT
def classify(image):
    if image is None:
        return "Vui lòng upload ảnh tế bào máu.", {}, "", ""

    prob_r, prob_s, prob_ens = predict(image)

    pred_idx  = int(np.argmax(prob_ens))
    pred_cls  = CLASSES[pred_idx]
    pred_conf = float(prob_ens[pred_idx]) * 100

    # Top-5 xác suất
    top5_idx  = np.argsort(prob_ens)[::-1][:5]
    top5_dict = {
        f"{CLASSES[i]} — {CLASS_NAMES[CLASSES[i]].split('—')[0].strip()}":
        float(prob_ens[i])
        for i in top5_idx
    }

    # Kết quả chính
    result_text = (
        f"## {pred_cls}\n\n"
        f"**{CLASS_NAMES[pred_cls]}**\n\n"
        f"**Độ tin cậy Ensemble:** {pred_conf:.1f}%\n"
    )
    if pred_cls in HARD_CLASSES:
        result_text += (
            f"\n*Lưu ý: {pred_cls} thuộc nhóm tế bào dòng tủy bào "
            f"(MY, MMY, PMY) có hình thái tương tự nhau. "
            f"Ensemble được thiết kế đặc biệt để cải thiện nhận dạng nhóm này.*"
        )

    # So sánh 3 mô hình
    r_pred = CLASSES[int(np.argmax(prob_r))]
    s_pred = CLASSES[int(np.argmax(prob_s))]
    r_conf = float(np.max(prob_r)) * 100
    s_conf = float(np.max(prob_s)) * 100

    compare_text = (
        f"### So sánh kết quả từng mô hình\n\n"
        f"| Mô hình | Dự đoán | Độ tin cậy | Test Accuracy |\n"
        f"|---|---|---|---|\n"
        f"| ResNet50  | {r_pred} | {r_conf:.1f}% | 94.01% |\n"
        f"| Swin-S    | {s_pred} | {s_conf:.1f}% | 94.49% |\n"
        f"| **Ensemble** | **{pred_cls}** | **{pred_conf:.1f}%** | **94.91%** |\n"
    )

    info_text = (
        f"**Lớp:** {pred_cls}\n\n"
        f"**Tên đầy đủ:** {CLASS_NAMES[pred_cls]}\n"
    )
    if pred_cls in HARD_CLASSES:
        info_text += "\n**Nhóm:** Dòng tủy bào (MY — MMY — PMY)"

    return result_text, top5_dict, compare_text, info_text

# GIAO DIỆN GRADIO
example_images = []
if os.path.exists(EXAMPLES_DIR):
    for f in sorted(os.listdir(EXAMPLES_DIR)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            example_images.append(os.path.join(EXAMPLES_DIR, f))

with gr.Blocks(
    title="Hệ Thống Phân Loại Tế Bào Máu",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown("""
    # Hệ Thống Phân Loại Tế Bào Máu
    ### Ensemble Learning: ResNet50 + Swin Transformer
    13 lớp tế bào &nbsp;|&nbsp; Accuracy: 94.91% &nbsp;|&nbsp; F1 Macro: 0.9317
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload ảnh tế bào máu (JPG/PNG)",
                height=300
            )
            submit_btn = gr.Button("Phân loại", variant="primary", size="lg")

            if example_images:
                gr.Examples(
                    examples=example_images,
                    inputs=image_input,
                    label="Ảnh mẫu"
                )

        with gr.Column(scale=1):
            result_md  = gr.Markdown("Upload ảnh và nhấn Phân loại để xem kết quả.")
            top5_label = gr.Label(label="Top-5 xác suất", num_top_classes=5)

    with gr.Row():
        compare_md = gr.Markdown()
        info_md    = gr.Markdown()

    gr.Markdown(f"""
    ---
    ResNet50: Accuracy {MODEL_STATS['ResNet50']['accuracy']}% | F1 {MODEL_STATS['ResNet50']['f1']}
    &nbsp;|&nbsp;
    Swin-S: Accuracy {MODEL_STATS['Swin-S']['accuracy']}% | F1 {MODEL_STATS['Swin-S']['f1']}
    &nbsp;|&nbsp;
    Ensemble: Accuracy {MODEL_STATS['Ensemble']['accuracy']}% | F1 {MODEL_STATS['Ensemble']['f1']}
    """)

    submit_btn.click(
        fn=classify,
        inputs=image_input,
        outputs=[result_md, top5_label, compare_md, info_md]
    )
    image_input.change(
        fn=classify,
        inputs=image_input,
        outputs=[result_md, top5_label, compare_md, info_md]
    )

# CHẠY
if __name__ == "__main__":
    print("=" * 50)
    print("Demo tại: http://localhost:7860")
    print("Nhấn Ctrl+C để dừng")
    print("=" * 50)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )