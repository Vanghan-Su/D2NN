# Dataset Card: 20261013-095617_mnist_1pct_pure_phase

## Source

- Dataset: MNIST
- Source root: `E:\codes\D2NN\data\MNIST\raw`
- Stored format: `(N, 1, 28, 28)` uint8 images and `(N,)` int64 labels
- Model preprocessing: bilinear resize to 400 x 400; per-sample unit L2 amplitude normalization in `TorchD2NN`

## Splits

| Split | Source split | Count | Indices |
|---|---|---:|---|
| Train | train | 600 | 0..599 |
| Validation | test | 100 | 0..99 |
| Test | test | 50 | See `split_manifest.json`; seed 20261013 |

The test set was sampled without replacement from official MNIST test indices 100..9999, excluding every validation index.

## Files

- `train_images.npy`, `train_labels.npy`
- `validation_images.npy`, `validation_labels.npy`
- `test_images.npy`, `test_labels.npy`
- `split_manifest.json`
