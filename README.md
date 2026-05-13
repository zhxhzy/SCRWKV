<h1 align="center">SCRWKV</h1>

<!-- <div align="center">
  <img src="figures/LOGO.png" width="220">
</div> -->

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




<div align="center">


    <img src="figures/Overview.png" width="850" alt="Overview">


</div>

Achieving pixel-level accurate segmentation of structural cracks across diverse scenarios remains a formidable challenge. Existing methods face significant bottlenecks in balancing crack topology modeling with computational efficiency, often failing to reconcile high segmentation quality with low resource demands. To address these limitations, we propose the Ultra-Compact Structure-Calibrated Vision RWKV (SCRWKV), a network that achieves high-precision modeling via a novel Structure-Field Encoder (SFE) backbone while maintaining linear complexity. The SFE integrates the Adaptive Multi-scale Cascaded Modulator (AMCM) to enhance texture representation and utilizes the Structure-Calibrated Insight Unit (SCIU) as its core engine. Specifically, the SCIU employs the Geometry-guided Bidirectional Structure Transform (GBST) to capture topological correlations and integrates the Dynamic Self-Calibrating Decay (DSCD) into Dy-WKV to suppress noise propagation. Furthermore, we introduce a lightweight Cross-Scale Harmonic Fusion (CSHF) decoder to achieve precise feature aggregation. Systematic evaluations on multiple benchmarks characterized by complex textures and severe interference demonstrate that SCRWKV, with only 1.22M parameters, significantly outperforms SOTA methods. Achieving an F1 score of 0.8428 and mIoU of 0.8512, the model confirms its robust potential for efficient real-world deployment.

## 🛠️ Installation

Create the environment:


```bash
conda create -n scrwkv python=3.9 -y
conda activate scrwkv
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 
pip install -U openmim
mim install mmcv-full==1.7.2
```
Simply install the required dependencies according to the project requirements.




---

## 📁 Dataset Preparation

Please create a new `data/` directory to store the datasets.

The expected dataset structure is:

- `data/`
  - `dataset_name_1/`

  - `dataset_name_2/`

You may need to modify the dataset path in the configuration or training script before running experiments.

---

## 🚀 Training

You can modify the parameters in the main.py file and run it with the following command:

```bash
python main.py
```

You can also specify your own arguments, for example:

```bash
python main.py \
  --dataset your_dataset \
  --
```

Please modify the command according to the arguments defined in `main.py`.

---

## 📊 Evaluation

You can also use checkpoints for inference with the following command:

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




## 🌍 Real-world Deployment

<div align="center">
  <img src="figures/real.gif" width="850">
</div>

---

## 🖼️ Visualization

Some qualitative visualization results are shown below.

<div align="center">
    <img src="figures/visual.png" width="850" alt="visual">
</div>

Visual comparison of typical crack segmentation results on the TUT dataset against 10 SOTA methods. Red boxes highlight
critical details, and green boxes mark misidentified regions.

> **Note:** Please replace the image path above with the exact filename in your `figures/` folder.

---

## 📌 Results




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
