from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from PIL import Image

from src.features import compare_images
from src.models import create_model, get_transforms


@dataclass
class Predictor:
    model: torch.nn.Module
    class_names: Dict[int, str]
    idx_to_class: Dict[int, str]
    transform: object
    device: torch.device


def load_name_mapping(mapping_json_path: Optional[str]) -> Dict[str, str]:
    if not mapping_json_path:
        return {}
    path = Path(mapping_json_path)
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}
    for key, values in data.items():
        if isinstance(values, list) and values:
            mapping[key] = values[0].strip()
        elif isinstance(values, str):
            mapping[key] = values.strip()
    return mapping


def load_predictor(checkpoint_path: str, mapping_json_path: Optional[str] = None) -> Predictor:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = ckpt["model_name"]
    class_to_idx = ckpt["class_to_idx"]
    input_size = int(ckpt.get("input_size", 224))

    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}

    model, _ = create_model(model_name=model_name, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    class_name_mapping = load_name_mapping(mapping_json_path)
    class_names: Dict[int, str] = {}
    for idx, folder_name in idx_to_class.items():
        raw_id = folder_name.replace("class_", "")
        class_names[idx] = class_name_mapping.get(raw_id, folder_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return Predictor(
        model=model,
        class_names=class_names,
        idx_to_class=idx_to_class,
        transform=get_transforms(input_size, is_train=False),
        device=device,
    )


def predict_image(predictor: Predictor, image_path: str) -> Tuple[int, float]:
    image = Image.open(image_path).convert("RGB")
    x = predictor.transform(image).unsqueeze(0).to(predictor.device)

    with torch.no_grad():
        logits = predictor.model(x)
        probs = torch.softmax(logits, dim=1)

    conf, pred_idx = probs.max(dim=1)
    return int(pred_idx.item()), float(conf.item())


def find_sample_image(dataset_root: str, class_folder: str) -> Optional[str]:
    candidates = [
        Path(dataset_root) / "train" / class_folder,
        Path(dataset_root) / "val" / class_folder,
        Path(dataset_root) / "test" / class_folder,
    ]
    for folder in candidates:
        if folder.exists() and folder.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                files = sorted(folder.glob(ext))
                if files:
                    return str(files[0])
    return None


def infer_with_verification(
    predictor: Predictor,
    image_path: str,
    dataset_root: str,
) -> Dict[str, object]:
    pred_idx, conf = predict_image(predictor, image_path)
    class_folder = predictor.idx_to_class[pred_idx]
    display_name = predictor.class_names[pred_idx]

    sample_image = find_sample_image(dataset_root, class_folder)
    compare_result = None
    if sample_image:
        compare_result = compare_images(sample_image, image_path)

    verdict = False
    score = 0
    if compare_result is not None:
        score = int(compare_result["score"])
        verdict = bool(compare_result["verdict"])

    return {
        "pred_idx": pred_idx,
        "class_folder": class_folder,
        "drug_name": display_name,
        "confidence": conf,
        "sample_image": sample_image,
        "compare": compare_result,
        "verdict": verdict,
        "score": score,
    }
