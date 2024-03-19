# 人臉辨識系列項目

這個倉庫包含了一系列進行人臉辨識的項目，使用了 PyTorch、OpenCV 並結合實時攝像頭偵測，實現了從基本的人臉辨識到具備資料儲存和時間記錄功能的應用。

## 項目列表

### Face_Recognize_v1

透過 PyTorch 和 OpenCV 來實作基本的人臉辨識功能。

### Face_Recognize_v2

在 `Face_Recognize_v1` 的基礎上加入資料儲存功能，能夠記憶人臉資料。

### OpenCV_with_cam

展示如何連接鏡頭實現實時人臉偵測。

### Face_Recognize_cam

結合人臉辨識和時間記錄功能，依照人名和月份來儲存時間資料，並創建了用戶介面，可用於打卡系統。

## 開始使用

### 所需套件

-numpy
-python
-opencv-python
-torch>=1.7.1
-facenet-pytorch
-pandas
-Pillow

### 安裝
若已完成環境設置，執行以下操作。

複製程式：

```bash
git clone https://github.com/cshuoc/face_Recognize_punch.git
```

安裝套件:
```bash
pip install -r requirements.txt
```

# PyTorch 環境設置

如何在機器上設置 PyTorch 環境，以便運行 PyTorch 相關的項目。

## 前提條件

系統需安裝以下軟體：

- pip (Python 包管理器)
- Anaconda 或 Miniconda
  
## 建立環境

開啟編譯器或Anaconda Prompt
用 conda 建立環境，環境名稱為your_env(可自行命名)，同時安裝python(若要指定python版本請改成python = 3.x)

```bash
conda create -n your_env python
ˋˋˋ

## 安裝 PyTorch

PyTorch 的安裝取決於系統配置（操作系統，是否需要 CUDA 支持等）。

### CPU 版本

```bash
pip install torch torchvision torchaudio
ˋˋˋ
### GPU 版本
支援 CUDA 11.3 的 Windows 系統，其他版本請至Pytorch官網查看

```bash
pip install torch torchvision torchaudio torchtext --extra-index-url https://download.pytorch.org/whl/cu113
ˋˋˋ

### 啟動環境

ˋˋˋbash
conda activate your_env
ˋˋˋ

### 驗證安裝是否安裝成功

```bash
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```


