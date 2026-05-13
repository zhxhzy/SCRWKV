<h1 align="center">SCRWKV</h1>

<div align="center">
  <img src="figures/LOGO.png" width="220">
</div>

<h3 align="center">
[ICML 2026] SCRWKV: Ultra-Compact Structure-Calibrated Vision-RWKV for Topological Crack Segmentation
</h3>

<p align="center">
  🖐😭🤚 🌟 If this work is useful to you, please give this repository a Star! 🌟 🖐😭🤚
</p>

<p align="center">
  <a href="https://arxiv.org/abs/你的arxiv编号">
    <img src="https://img.shields.io/badge/arXiv-你的编号-b31b1b.svg">
  </a>
  <a href="你的论文链接">
    <img src="https://img.shields.io/badge/Paper-ICML%202026-blue">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg">
  </a>
  <a href="https://github.com/zhxhzy/SCRWKV">
    <img src="https://img.shields.io/badge/Code-GitHub-black">
  </a>
</p>

---

## 📜 News

- **2026-05-01:** 🎉 We are delighted to announce that our paper has been accepted by **ICML 2026**! 🖐😭🤚
- **2026-XX-XX:** 📦 The code has been publicly released.
- **2026-XX-XX:** 📝 The preprint has been posted on arXiv.

---

## 🔥 Overview

This repository contains the official implementation of:

> **SCRWKV: Ultra-Compact Structure-Calibrated Vision-RWKV for Topological Crack Segmentation**  
> International Conference on Machine Learning, 2026

<div align="center">
  <!-- 外层用 <a> 标签指向你的 PDF 文件 -->
  <a href="figures/Overview.pdf">
    <!-- 里层用 <img> 标签显示转好的预览图 -->
    <img src="figures/Overview.png" width="850" alt="Overview">
  </a>
  <br>
  <sup>*Click the image to view the high-resolution PDF*</sup>
</div>

---

## 🧩 Project Structure

The repository is organized as follows:

- `datasets/`: dataset directory
- `eval/`: evaluation scripts and related files
- `figures/`: figures used in this README
  - `LOGO.png`
  - `Overview.png`
  - `Real-world_Deployment.gif`
  - `Visualization_on_publicly_available_*.png`
- `mmcls/`: classification-related modules
- `models/`: model implementations
- `results/`: experimental results
- `util/`: utility functions
- `engine.py`: training and inference engine
- `eval_compute.py`: metric computation script
- `main.py`: main training script
- `test.py`: testing and inference script
- `LICENSE`: open-source license
- `README.md`: project documentation

---

## 🛠️ Installation

Clone this repository:

```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
```

Create the environment:

```bash
conda create -n scrwkv python=3.10 -y
conda activate scrwkv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If there is no `requirements.txt` yet, please install the core dependencies manually according to your environment.

For example:

```bash
pip install torch torchvision numpy opencv-python tqdm matplotlib scikit-learn
```

---

## 📁 Dataset Preparation

Please place the datasets under the `datasets/` directory.

The expected dataset structure is:

- `datasets/`
  - `dataset_name_1/`
    - `train/`
    - `val/`
    - `test/`
  - `dataset_name_2/`
    - `train/`
    - `val/`
    - `test/`

You may need to modify the dataset path in the configuration or training script before running experiments.

---

## 🚀 Training

Run the training script:

```bash
python main.py
```

You can also specify your own arguments, for example:

```bash
python main.py \
  --dataset your_dataset \
  --model scrwkv \
  --batch-size 8 \
  --epochs 100
```

Please modify the command according to the arguments defined in `main.py`.

---

## 📊 Evaluation

Run evaluation with:

```bash
python test.py
```

Compute evaluation metrics with:

```bash
python eval_compute.py
```

The evaluation-related files are located in the `eval/` directory.

The results will be saved in the `results/` directory.

---

## 🏗️ Model

The model implementations are located in the `models/` directory.

The main model components and utilities are organized as follows:

- `models/`
- `util/`
- `mmcls/`
- `engine.py`
- `main.py`
- `test.py`

---

## 🌍 Real-world Deployment

<div align="center">
  <img src="figures/real" width="850">
</div>

---

## 🖼️ Visualization

Some qualitative visualization results are shown below.

<div align="center">
  <img src="figures/visual.pdf" width="850">
</div>

> **Note:** Please replace the image path above with the exact filename in your `figures/` folder.

---

## 📌 Results

| Method | Dataset | Metric 1 | Metric 2 | Metric 3 |
|---|---|---:|---:|---:|
| Baseline | Dataset A | 00.00 | 00.00 | 00.00 |
| SCRWKV | Dataset A | **00.00** | **00.00** | **00.00** |

More results can be found in the `results/` directory.

---

## 📜 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{scrwkv2026,
  title={SCRWKV: Ultra-Compact Structure-Calibrated Vision-RWKV for Topological Crack Segmentation},
  author={Author A and Author B and Author C},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

---

## 📄 License

This project is released under the Apache 2.0 License.

Please see the [LICENSE](LICENSE) file for more details.

---

## 🙏 Acknowledgements

We sincerely thank the authors of the open-source projects and datasets that made this work possible.
