# LUNGLENS 🫁
Chest X-ray classification into four classes (COVID, Lung_Opacity,
Normal, Viral_Pneumonia) using DenseNet121 transfer learning.
**Research demo only. Not a medical diagnosis.**

## Team
- Kenas Jaison: model training, evaluation, integration
- [Member 2 name]: data preparation and evaluation

## Dataset
Kaggle COVID-19 Radiography Database (Tawsifur Rahman), 21,165
chest X-rays: COVID 3,616 | Lung_Opacity 6,012 | Normal 10,192 |
Viral Pneumonia 1,345. Stratified 70/15/15 split (seed 42).

## Method
1. DenseNet121 pre-trained on ImageNet, base frozen (baseline)
2. Class weights to handle the class imbalance
3. Fine-tuning of the last 30 layers with a low learning rate
Input: 224x224 RGB, raw 0-255 pixels (scaling is inside the model).

## Results (held-out test set, 3,175 images)
| Class | Precision | Recall |
|---|---|---|
| COVID | 0.812 | 0.887 |
| Lung_Opacity | 0.853 | 0.854 |
| Normal | 0.915 | 0.870 |
| Viral_Pneumonia | 0.868 | 0.975 |

Accuracy 0.875, macro-average recall 0.897.
Baseline COVID recall was 0.683; class weights and fine-tuning raised
it to 0.887.

## Files
- `model/`: training notebook with all outputs
- `data_eval/`: evaluation work (charts, notes)
- `app/`: demo app
- Trained model (`lung_model.keras`): [Google Drive link]

## Limitations
One dataset, one random image-level split (possible near-duplicate
images across splits), no clinical validation. Not for medical use.
