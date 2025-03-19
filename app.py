import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tkinter import filedialog

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

COCO_LABELS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
]


def detect_objects(image_path, threshold=0.5):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        return

    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    try:
        with torch.no_grad():
            predictions = model(image_tensor)[0]

        if not predictions["boxes"].size(0):
            print("Модель не нашла объекты на изображении.")
            return
    except Exception as e:
        print(f"Ошибка при получении предсказаний: {e}")
        return

    try:
        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.imshow(image)

        for i in range(len(predictions["boxes"])):
            score = predictions["scores"][i].item()

            if score >= threshold:
                box = predictions["boxes"][i].tolist()

                label_index = predictions["labels"][i].item()
                if label_index >= len(COCO_LABELS):
                    label = "unknown"
                else:
                    label = COCO_LABELS[label_index]

                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(box[0], box[1] - 5, f"{label} ({score:.2f})", color="red", fontsize=12)
    except Exception as e:
        print(f"Ошибка при визуализации: {e}")
        return

    plt.show()

file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
if file_path:
    detect_objects(file_path)