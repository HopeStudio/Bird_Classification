import argparse
from pathlib import Path

import torch
from torchvision import models


ROOT_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=ROOT_DIR / "models/model20240824.pth")
    parser.add_argument("--output-path", type=Path, default=ROOT_DIR / "models/model20240824.onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))

    model = models.resnet34(num_classes=11000)
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        args.output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch_size")},),
        opset_version=args.opset,
        dynamo=True,
        external_data=False,
    )

    print(f"Exported ONNX model to: {args.output_path}")


if __name__ == "__main__":
    main()
