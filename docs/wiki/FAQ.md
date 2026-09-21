# FAQ

**Why is accuracy 98.2% when stroke prediction is a genuinely hard clinical problem?**
The reported metrics come from oversampling the minority class before splitting into train/test, so some information from the same duplicated/near-duplicated minority-class samples can appear on both sides of the split. This is a realistic result for the pipeline as implemented, not a validated clinical benchmark — see [Methodology](Methodology) for the full caveat.

**Can I use this model for real medical decisions?**
No. This is an educational/portfolio project built on a public Kaggle dataset, not a validated clinical tool. Any real diagnostic use would require a properly validated pipeline (oversampling only on the training fold, external validation, clinical review).

**Where does the trained model come from?**
`notebook/xgb_model.pkl` is produced by running `Equipe_NotebookV1.ipynb` end to end. It's committed to the repo so the Streamlit demo works out of the box without retraining.

**What does the Streamlit demo actually do?**
`notebook/app.py` loads `xgb_model.pkl` and exposes a form for entering a patient's profile (age, glucose level, BMI, hypertension, etc.), then runs it through the same preprocessing/encoding used in training to produce a stroke-risk prediction.

**Why XGBoost specifically?**
It handles the mixed categorical/numeric feature set well and is a common strong baseline for structured/tabular classification tasks like this one.

**Does CI retrain the model or verify the 98.2% figure?**
No — CI only checks notebook structural integrity and that dependencies install cleanly. It doesn't execute the training cells or re-verify the metrics.
