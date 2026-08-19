# Aerial Fire Detection using Computer Vision and Drone

Detects fire in aerial/drone footage with [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/),
annotated with [Supervision](https://supervision.roboflow.com/).

> **Updated August 2026.** I've modernised this project end to end. The original
> notebooks were written against early-2023 libraries and no longer ran — and two of
> them wouldn't even render on GitHub or open in Colab. Everything below now runs on
> current releases, and I've added [`main.ipynb`](main.ipynb) — an updated version of the
> [Roboflow blog post](https://blog.roboflow.com/aerial-fire-detection/) this project
> accompanies. See [What I changed](#what-i-changed).

## Notebooks

**Start with [`main.ipynb`](main.ipynb) — an updated version of the Roboflow blog post
[Aerial Fire Detection with Drone Imagery and Computer Vision](https://blog.roboflow.com/aerial-fire-detection/).**

It covers the same ground as the original article — why aerial fire detection, how the
system works end to end, dataset preparation, labelling, training, and inference on
images and video — but with the code updated to libraries that still run, and with each
section handing you off to the relevant runnable notebook at the right point.

The article dates from September 2023. Some of its code still runs fine, some is merely
dated — and the annotation and tracking sections no longer run at all. `main.ipynb` flags
each of those points as it reaches them, rather than leaving you to find out by running it.

| Notebook | Purpose | |
|---|---|---|
| `main.ipynb` | **Start here** — updated version of the [Roboflow blog post](https://blog.roboflow.com/aerial-fire-detection/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakkzz/Fire-Detection-Drone/blob/main/main.ipynb) |
| `drone_fire_detection_yolo26.ipynb` | Train the model on the Roboflow dataset | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakkzz/Fire-Detection-Drone/blob/main/drone_fire_detection_yolo26.ipynb) |
| `Supervision_Image_Inferencing.ipynb` | Run the trained model on a single image | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakkzz/Fire-Detection-Drone/blob/main/Supervision_Image_Inferencing.ipynb) |
| `Supervision_Video_Inferencing.ipynb` | Detection + ByteTrack tracking on video | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jakkzz/Fire-Detection-Drone/blob/main/Supervision_Video_Inferencing.ipynb) |

Run the three numbered notebooks in that order. Training exports a `best.pt` checkpoint
that both inference notebooks expect you to upload — it isn't committed here. Use a GPU runtime
(`Runtime` → `Change runtime type` → **T4 GPU**); the image notebook also works on CPU.

Sample inputs `fire_image.png` and `fire.mp4` live in this repo and are fetched
automatically by the inference notebooks.

## Versions

| Package | Version |
|---|---|
| `ultralytics` | `>=8.4.122` |
| `supervision` | `>=0.30.0` |
| `roboflow` | `>=1.4.1` |
| Model | `yolo26m.pt` (swap to `n`/`s`/`l`/`x` in the training notebook) |

Version floors rather than exact pins, so pip can still resolve against whatever
PyTorch build Colab ships that week.

## What I changed

**The notebooks wouldn't open.** Two distinct causes:

- `Supervision_Video_Inferencing.ipynb` carried a `metadata.widgets` block whose
  `application/vnd.jupyter.widget-state+json` entry was missing its required `state`
  key — the exact condition behind GitHub's *"Invalid Notebook"* error.
- The training notebook was 1.68 MB, over GitHub's ~1 MB rich-render ceiling. 1.67 MB
  of that was embedded base64 output images.

All three were `nbformat 4.0` without cell IDs. They're now 4.5 with IDs, outputs
stripped, and clean metadata — down to roughly 8 KB, 5 KB and 10 KB.

**The code was dead, not just dated.**

- Upgraded to **YOLO26**, which is end-to-end and NMS-free, from `yolov8m.pt`.
- `ultralytics==8.0.20` (January 2023) was pinned; that pin no longer builds on current
  Colab. `supervision==0.1.0` used module paths (`supervision.tools.detections`,
  `supervision.draw.color`) that no longer exist.
- `BoxAnnotator(..., labels=...)` was removed in supervision 0.22 — drawing is now split
  across `BoxAnnotator` and `LabelAnnotator`.
- **Tracking was the worst of it.** The old notebook cloned
  [ByteTrack](https://github.com/ifzhang/ByteTrack), built YOLOX from source, and
  installed `onemetric` / `cython_bbox`, with two `sed` patches for long-closed issues —
  a toolchain that no longer builds. ByteTrack ships inside Ultralytics now, so
  `model.track(persist=True, tracker="bytetrack.yaml")` replaces all of it. Worth noting:
  the old render loop constructed a `BYTETracker` and never actually used it, so the
  "tracking" output was really just untracked per-frame detections.
- ByteTrack is fed low-confidence detections on purpose — recovering weak boxes by
  matching them to live tracks is where much of its accuracy comes from — so the
  confidence filter is applied after tracking rather than starving the tracker.
- Removed hardcoded `runs/detect/train` paths in favour of `results.save_dir`, since
  Ultralytics increments to `train2`, `train3` on reruns.
- The Roboflow key is read from Colab Secrets / environment instead of a literal
  `api_key="YOUR_API_KEY"` in a cell.
- Added an ffmpeg H.264 re-encode step. Supervision writes `mp4v`, which browsers won't
  decode, so the result video never played back inline.

Outputs were stripped, so the notebooks no longer display training curves inline on
GitHub — unavoidable for the oversized file, and those outputs were stale against the
rewritten code anyway.

## Dataset

Alireza Shamsoshoara, Fatemeh Afghah, Abolfazl Razi, Liming Zheng, Peter Fulé, Erik Blasch,
November 19, 2020, "The FLAME dataset: Aerial Imagery Pile burn detection using drones (UAVs)",
IEEE Dataport, doi: <https://dx.doi.org/10.21227/qad6-r683>.

Training pulls the labelled version from Roboflow, which needs a free
[API key](https://app.roboflow.com/settings/api). Store it in Colab under
🔑 **Secrets** as `ROBOFLOW_API_KEY` with *Notebook access* enabled.

## Credit

Forked from [tim3in/Fire-Detection-Drone](https://github.com/tim3in/Fire-Detection-Drone).

`main.ipynb` is an updated version of
[Aerial Fire Detection with Drone Imagery and Computer Vision](https://blog.roboflow.com/aerial-fire-detection/)
by Timothy M. on the Roboflow Blog (Sep 19, 2023). It covers the same material, rewritten
as this repo's own walkthrough rather than reproduced, with the code brought up to date
and the outdated snippets called out where they appear.

> Timothy M. (Sep 19, 2023). Aerial Fire Detection with Drone Imagery and Computer
> Vision. Roboflow Blog: https://blog.roboflow.com/aerial-fire-detection/
