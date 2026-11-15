# Face Attractiveness Predictor

Predicts how attractive a face looks, either to an average rater (trained on
[SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release)) or to one specific person (trained on their
own ratings).

A successor of [CNN-ER_for_FBP](https://github.com/yaskovdev/CNN-ER_for_FBP). That project fine-tuned an ensemble of
six large CNNs, and its predictions on real photos often disagreed with my own judgement. The likely reasons:

* the training faces were cropped using the dataset's hand-labelled landmarks, but photos at inference time went through
  a different detector (Haar + LBF), so the model saw differently framed faces;
* SCUT-FBP5500 contains studio-like frontal photos, mostly of Asian faces, rated by 60 Chinese students, so the model
  learns *their* average taste, not mine;
* individual taste explains a substantial part of attractiveness ratings, so only a model trained on my ratings can
  agree with me.

## Approach

1. **One alignment for everything.** OpenCV's YuNet detector finds 5 facial landmarks, and a similarity transform maps
   them to a fixed template (`attractiveness/face_alignment.py`). Dataset faces and new photos are processed
   identically.
2. **Frozen pretrained features + ridge regression.** A large pretrained image encoder (CLIP ViT-L/14 or DINOv2
   ViT-L/14) turns each aligned face into a feature vector; a ridge regression on top predicts the score. This is
   cheap to train, needs few labelled faces and is a good fit for a personal dataset of a few hundred ratings.
3. **Personal ratings.** `personal_ratings/` contains tooling for collecting my own ratings.

Photos and ratings are kept outside the repository and must never be committed.

## Results on SCUT-FBP5500 (official 60/40 split, 2200 test faces)

| Features                      | Pearson | MAE    | RMSE   |
|-------------------------------|---------|--------|--------|
| CLIP ViT-L/14                 | 0.9417  | 0.1777 | 0.2283 |
| DINOv2 ViT-L/14               | 0.8912  | 0.2386 | 0.3080 |
| CLIP ViT-L/14 + DINOv2        | 0.9446  | 0.1721 | 0.2227 |

CLIP alone is used by default: the concatenation is barely better but twice as slow. Training the ridge regression takes
seconds; extracting the features of all 5500 faces takes about a minute on an RTX 3080 Ti.

## Setup (Windows, NVIDIA GPU)

```powershell
python -m venv .venv
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip install -r requirements.txt
```

## Usage

Below, `$scut` is the `SCUT-FBP5500_v2` folder of the unpacked dataset, with `train_test_files.zip` extracted in place.

```powershell
# 1. Align all faces
python -m attractiveness.align_images --input "$scut\Images" --output C:\dev\fbp_data\aligned\scut

# 2. Extract features (repeat for every backbone you want to compare)
python -m attractiveness.extract_features --images C:\dev\fbp_data\aligned\scut --backbone clip_vit_l14 --output features\scut_clip_vit_l14.npz

# 3. Evaluate on the official 60/40 split
$split = "$scut\train_test_files\split_of_60%training and 40%testing"
python -m attractiveness.evaluate --features features\scut_clip_vit_l14.npz --ratings "$scut\train_test_files\All_labels.txt" --train-list "$split\train.txt" --test-list "$split\test.txt"

# 4. Train the final model on all faces and score your own photos
python -m attractiveness.train_model --features features\scut_clip_vit_l14.npz --ratings "$scut\train_test_files\All_labels.txt" --output models\scut_clip_vit_l14.joblib
python -m attractiveness.predict --model models\scut_clip_vit_l14.joblib --save-aligned aligned_output photo.jpg
```

`evaluate` without `--train-list`/`--test-list` runs k-fold cross-validation, which is what to use on personal ratings
(an `image,score` CSV). Add `--reference-model models\scut_clip_vit_l14.joblib` to see how well the SCUT model agrees with them.

### Rating photos yourself

```powershell
python -m personal_ratings.rate_photos --images C:\dev\fbp_data\photofeeler_images --output C:\dev\fbp_data\personal_ratings.csv
```

Shows the photos one by one; rate the face on a 1-10 scale with keys 1-9 and 0 (= 10), S skips, B goes back, Q quits.
Progress is saved after every photo, so rating can be resumed at any time. Use the whole scale.

### Capturing Photofeeler votes

```powershell
mitmdump -s personal_ratings\photofeeler_capture_addon.py
```

Saves the photos shown while voting to `captured_images/` and logs vote requests to `captured_events.jsonl`.
