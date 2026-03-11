import os
import cv2
import yaml
import argparse
import numpy as np
from PIL import Image
import albumentations as A

parser= argparse.ArgumentParser()
parser.add_argument("--file", type=str)
args = parser.parse_args()

log_file = open("output.txt", "w", encoding="utf-8")
def log(message):
    print(message)
    log_file.write(message + "\n")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

inter_map = {
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4
}

def parse_transform(transform_dict, h, w):
    t_type = transform_dict["type"]
    params = dict(transform_dict.get("params", {})) 

    if t_type == "Resize":
        if "height_scale" in params:
            params["height"] = int(h * params.pop("height_scale"))
        elif params.get("height") == "original":
            params["height"] = h

        if "width_scale" in params:
            params["width"] = int(w * params.pop("width_scale"))
        elif params.get("width") == "original":
            params["width"] = w

        if "interpolation" in params:
            interp_name = params["interpolation"]
            params["interpolation"] = inter_map.get(interp_name, cv2.INTER_LINEAR)

    if hasattr(A, t_type):
        albumentation_class = getattr(A, t_type)
        return albumentation_class(**params)
    else:
        raise ValueError(f"[!] Transformare necunoscută: {t_type}")

def build_effect(effect_config, h, w):
    if effect_config["type"] == "Compose":
        transforms = [parse_transform(t, h, w) for t in effect_config["transforms"]]
        return A.Compose(transforms)
    else:
        return parse_transform(effect_config, h, w)

def extract_param_values(params: dict, order: list):
    values = []
    for key in order:
        val = params.get(key)
        if isinstance(val, list):
            values.extend(str(v).replace('.', 'p') for v in val)
        else:
            values.append(str(val).replace('.', 'p'))
    return values

def format_param_string(effect_name, effect_config):
    if effect_config["type"] == "Compose":
        values = []
        for t in effect_config["transforms"]:
            param_vals = extract_param_values(t.get("params", {}), t.get("param_order", []))
            values.extend(param_vals)
        return f"{effect_name}_{'_'.join(values)}"
    else:
        vals = extract_param_values(effect_config.get("params", {}), effect_config.get("param_order", []))
        return f"{effect_name}_{'_'.join(vals)}"

input_dir = "input"

if args.file:
    base_name = os.path.splitext(os.path.basename(args.file))[0]
    output_dir = os.path.join("output", f"{base_name}_output")
    image_files = [args.file]
else:
    output_dir = "output"
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]

os.makedirs(output_dir, exist_ok=True)

min_h, min_w = 80, 224

for img_path in image_files:
    img_file = os.path.basename(img_path)
    try:
        image = np.array(Image.open(img_path).convert("RGB"))
    except Exception as e:
        log(f"[!] Eroare la citirea fișierului {img_file}: {e}")
        continue

    h, w = image.shape[:2]

    if h < min_h or w < min_w:
        log(f"[i] Redimensionare: {img_file} de la ({w}x{h}) la ({min_w}x{min_h})")
        image = cv2.resize(image, (min_w, min_h), interpolation=cv2.INTER_CUBIC)
        h, w = image.shape[:2]

    for effect_name, effect_config in config["effects"].items():
        try:
            transform = build_effect(effect_config, h, w)
            augmented = transform(image=image)["image"]
        except Exception as e:
            log(f"[!] Eroare la aplicarea efectului {effect_name} pe {img_file}: {e}")
            continue

        param_string = format_param_string(effect_name, effect_config)
        base_name, ext = os.path.splitext(os.path.basename(img_file))
        filename = f"{base_name}_{param_string}{ext}"
        output_path = os.path.join(output_dir, filename)

        try:
            cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
            log(f"[✓] Salvat: {output_path}")
        except Exception as e:
            log(f"[!] Eroare la salvare {output_path}: {e}")

log_file.close()
