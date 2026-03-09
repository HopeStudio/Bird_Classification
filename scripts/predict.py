import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models/model20240824.pth"
CSV_PATH = ROOT_DIR / "db/bird_info.csv"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path, device):
    model = models.resnet34(num_classes=11000)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def load_bird_info(csv_path):
    bird_info = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            class_id = row["model_class_id"].strip()
            if class_id:
                bird_info[int(class_id)] = row
    return bird_info


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def prepare_image(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return image, tensor


def predict_normal(model, image_tensor, num_classes):
    with torch.no_grad():
        output = model(image_tensor)[0][:num_classes]
        return torch.softmax(output, dim=0)


def predict_tta(model, image, transform, device, num_classes):
    image_1 = transform(image).unsqueeze(0).to(device)
    image_2 = transform(image.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(device)

    with torch.no_grad():
        output_1 = model(image_1)[0][:num_classes]
        output_2 = model(image_2)[0][:num_classes]
        return torch.softmax((output_1 + output_2) / 2, dim=0)


def get_results(probs, bird_info, top_k):
    top_probs, top_ids = torch.topk(probs, top_k)
    results = []

    for class_id, score in zip(top_ids.tolist(), top_probs.tolist()):
        info = bird_info.get(class_id, {
            "model_class_id": str(class_id),
            "chinese_simplified": "未知",
            "english_name": "Unknown",
            "scientific_name": "",
            "short_description_zh": "",
        })
        results.append({
            "class_id": class_id,
            "confidence": score * 100,
            "info": info,
        })

    return results


def print_results(results, mode):
    print(f"预测模式: {mode}")
    print()

    best = results[0]
    info = best["info"]
    print("最可能的结果:")
    print(f"model_class_id: {info['model_class_id']}")
    print(f"中文名: {info['chinese_simplified']}")
    print(f"英文名: {info['english_name']}")
    print(f"学名: {info['scientific_name']}")
    print(f"置信度: {best['confidence']:.2f}%")
    print(f"简介: {info['short_description_zh']}")
    print()

    print("候选结果:")
    for index, item in enumerate(results, start=1):
        item_info = item["info"]
        print(
            f"{index}. "
            f"{item_info['chinese_simplified']} / "
            f"{item_info['english_name']} / "
            f"{item_info['scientific_name']} / "
            f"{item['confidence']:.2f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="鸟类图片分类")
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument(
        "--mode",
        choices=["normal", "tta"],
        default="normal",
        help="normal=普通预测，tta=翻转增强预测",
    )
    parser.add_argument("--top-k", type=int, default=5, help="显示前几个结果")
    args = parser.parse_args()

    device = get_device()
    model = load_model(MODEL_PATH, device)
    bird_info = load_bird_info(CSV_PATH)
    num_classes = max(bird_info.keys()) + 1
    transform = get_transform()
    image, image_tensor = prepare_image(args.image, transform, device)

    if args.mode == "tta":
        probs = predict_tta(model, image, transform, device, num_classes)
    else:
        probs = predict_normal(model, image_tensor, num_classes)

    print_results(get_results(probs, bird_info, args.top_k), args.mode)


if __name__ == "__main__":
    main()
