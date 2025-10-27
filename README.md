# 🏷️ Experiment Log: Exp-Image-Classifier-007

**モデル名:** ResNet50_TransferLearning_V2  
**担当者:** Gemini AI (ジェミニ)  
**実行日:** 2025-10-27

### 目的
新しいデータ拡張手法（Mixup）の導入による汎化性能の評価。

---

## 💻 コードと環境

| 項目 | 詳細 |
| :--- | :--- |
| **Git Commit Hash** | `f8a2e1b9c4d6f0a3` |
| **コードへのリンク** | [この実験のコードの状態を見る](https://github.com/your-org/your-repo/tree/f8a2e1b9c4d6f0a3)  
| **実行コマンド** | `python train.py --model resnet50 --mixup --lr 0.001 --epochs 10`  

---

## 💾 データセット情報

**データセット名:** TinyImageNet_Cleaned_v1.1  
**NFSパス:** `/mnt/nfs_data/datasets/tiny_imagenet/v1.1/`

---

## ⚙️ ハイパーパラメータ

| パラメータ | 設定値 |
| :--- | :--- |
| ベースモデル | `torchvision.models.resnet50` (ImageNet事前学習済み) |
| 学習率 (LR) | 1e-3 |
| エポック数 | 10 |
| バッチサイズ | 128 |
| データ拡張 | RandomCrop, RandomFlip, **Mixup (Alpha 0.2)** |

---

## 📈 結果とモデルの場所

### 評価メトリクス

| メトリクス | 値 |
| :--- | :--- |
| **最高検証精度** | **81.54%** (エポック 8) |
| 最終検証損失 | 0.893 |

### 保存場所

* **モデルのNFSパス**: `/mnt/nfs_models/image_cls/Exp-007/resnet50_e08_81_54.pth`
* **学習曲線グラフ**:  (NFSから出力したグラフ画像へのリンクをここに記載することが多いです)

### 考察

Mixupデータ拡張の導入により、ベースライン（79.8%）と比較して約1.7%の精度向上を確認。次の実験では学習率スケジューラーをCosine Annealingに変更することを検討する。
