# LPSR-Degradation-Toolkit
This repository provides a systematic methodology for creating challenging license plate datasets. It allows for the programmatic generation of complex image artifacts, moving beyond simple noise to model environmental and optical conditions such as weather interference and motion blur. Focused on the creation and augmentation of training data for License Plate Super-Resolution (LPSR) models, the included module generates paired HR (High Resolution) and LR (Low Resolution) images by applying diverse image artifacts to original license plate photos.

## Repository content

* **`image_distortion_set.py`**: The main script used to automatically apply distortions using the [Albumentations](https://explore.albumentations.ai/) library.
* **`config.yaml`**: The settings file where you can configure the effects (motion blur, weather conditions, distance, perspective, etc.).
* **`source_images.txt`**: A text file containing direct links to the original high-resolution license plate images used for the dataset.

## Input Requirements 

The images referenced in **`source_images.txt`**  were sourced from [PlatesMania](https://platesmania.com/), a specialized database of vehicle plates from actual traffic. To replicate the dataset or use the script effectively, please note the following pre-processing steps applied to the source data:

* ***Plate Extraction & Filtering***: Regions of interest (ROI) were localized and cropped using a [Roboflow](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e) detector (97.2% mAP@50) with a confidence threshold > 70% and a validated aspect ratio between 3.0 and 6.0.

* ***Resolution & Upsampling***: The script enforces a minimum resolution threshold of 224 × 80 pixels. Any images falling below this limit are automatically upsampled using cubic interpolation prior to the degradation phase to ensure the stable execution of augmentation kernels like rain and snow.

<p align="center">
  <img src="media/samples.jpg" width="400" alt="Original PlatesMania Samples">
  <img src="media/samples-cropped.jpg" width="400" alt="Extracted ROI Samples">
  <br>
  <em>Left: Full vehicle captures from PlatesMania (Source) Right: Extracted & Filtered ROI crops (Input)</em>
</p>

## Configuration

The toolkit is modular, with all distortion parameters managed through **`config.yaml`**. Each effect can be toggled and fine-tuned by adjusting its specific values and execution probability (`p`). For a comprehensive list of all available image augmentations and their technical parameters, please refer to the official [Albumentations Documentation](https://albumentations.ai/docs/).

### Available Distortion Effects

To better understand the degradation effects, the visual samples in the table below are generated from the following high-resolution crop:

<p align="center">
  <img src="media/B694SOF_plate_crop.jpg" width="200" alt="Ground Truth Reference">
  <br>
  <em>Original HR License Plate (Ground Truth)</em>
</p>

| Effect ID | Description | Visual Sample | Albumentations Transforms |
|:---|:---|:---|:--- |
| **`MOTION`** | Motion blur simulation | <img src="media/B694SOF_plate_crop_MOTION_7_15_1p0.jpg" width="150"> | `MotionBlur` | 
| **`DISTANCE`** | Simulates long-range capture | <img src="media/B694SOF_plate_crop_DISTANCE_0p8_0p8_INTER_AREA_original_original_INTER_LINEAR_6_10_1p0_0p2_0p25_1p0.jpg" width="150"> | `Resize` + `Downscale` + `GaussianBlur` |
| **`SKEW`** | Perspective distortion | <img src="media/B694SOF_plate_crop_SKEW_0p08_0p35_True_True_1p0.jpg" width="150"> | `Perspective` |
| **`OBSTRUCTION`** | Partial plate occlusions | <img src="media/B694SOF_plate_crop_OBSTRUCTION_2_5_0p1_0p2_0p1_0p2_0.jpg" width="150"> | `CoarseDropout` |
| **`DAY_RAIN`** | Rain in daylight conditions | <img src="media/B694SOF_plate_crop_DAY_RAIN_20_1_200_200_200_9_0p97_torrential_1p0_5_1p0.jpg" width="150"> | `RandomRain` + `MotionBlur` |
| **`DAY_SNOW`** | Snow in daylight conditions | <img src="media/B694SOF_plate_crop_DAY_SNOW_0p45_0p48_2_bleach_1p0_0p25_0p28_0p3_texture_1p0.jpg" width="150"> | `RandomSnow` (bleach + texture) |
| **`NIGHT_RAIN`** | Night-time rain simulation | <img src="media/B694SOF_plate_crop_NIGHT_RAIN_15_1_230_230_230_4_0p72_torrential_1p0_-0p5_-0p2_0p3_0p6_1p0_0p1_0p2_0_0_True_1_0p5_1p0.jpg" width="150"> | `RandomRain` + `Brightness/Contrast` + `GaussNoise` + `ToGray` |
| **`NIGHT_SNOW`** | Night-time snow simulation | <img src="media/B694SOF_plate_crop_NIGHT_SNOW_0p3_0p35_1p0_texture_1p0_0p3_0p35_1p0_bleach_1p0_-0p5_-0p4_0p3_0p6_1p0_0p1_0p2_0_0_True_1_0p5_1p0.jpg" width="150"> | `RandomSnow` + `Brightness/Contrast` + `GaussNoise` + `ToGray` |
| **`OVEREXPOSURE`** | Strong light overexposure | <img src="media/B694SOF_plate_crop_OVEREXPOSURE_0p4_0p7_-0p4_-0p2_1p0_0_-20_40_1p0_3_0p2.jpg" width="150"> | `RandomBrightnessContrast` + `HueSaturationValue` + `MotionBlur` |

## Usage

### Requirements
Install the necessary dependencies:

```bash
pip install opencv-python pillow pyyaml albumentations numpy
```
### Full Dataset Processing

To process all images in the input/ directory:

```bash
python image_distortion_set.py
```

This command will:
1. Read all `.jpg` images from the `input/` folder.
2. Apply every effect defined in `config.yaml`.
3. Save the processed images to the `output/` folder.
4. Generate an `output.txt` file with detailed execution logs.

### Individual Image Processing
To process a single specific image:

```bash
python image_distortion_set.py --file path/to/image.jpg
```

### Naming Convention
Generated files include the specific parameters used during processing within the filename:


Format: `[original_name]_[EFFECT]_[param1]_[param2]_..._[paramN].jpg`

Exemplu:
```
AG24EGK_MOTION_7_15_1p0.jpg
│       │      │  │   │
│       │      │  │   └─ p=1.0 (Probability)
│       │      │  └───── blur_limit max=15
│       │      └──────── blur_limit min=7
│       └─────────────── Effect Type
└─────────────────────── Original Image Name
```

### Directory Structure
```
LPSR-Degradation-Toolkit/
├── config.yaml                    # Effect configurations
├── image_distortion_set.py        # Main distortion script
├── input/                         # Original (HR) images directory
│   ├── plate001.jpg
│   ├── plate002.jpg
│   └── ...
└── output/                        # Processed (LR) images directory
    ├── plate001_MOTION_7_15_1p0.jpg
    ├── plate001_DISTANCE_...jpg
    └── ...
```
