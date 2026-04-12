# FairHire

## Kaggle resume samples (optional)

The Streamlit UI can load real rows from
[snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
via `kagglehub`. Teammates only need this if they use that expander in `app.py`.

1. **Install** (included in `requirements.txt`): `pip install kagglehub`
2. **Kaggle API credentials** — pick one:
   - **File (recommended):** Kaggle → your avatar → **Account** → **Create New API Token**. Save the downloaded `kaggle.json` as `~/.kaggle/kaggle.json` (create the `.kaggle` folder if needed). **Never commit** `kaggle.json`.
   - **Environment variables:** set `KAGGLE_USERNAME` and `KAGGLE_KEY` to the values shown on the same API page (same secret as in the JSON).
3. **Dataset access:** open the dataset page on Kaggle once and accept any **terms / “Join”** prompt if Kaggle asks; otherwise downloads can fail with an auth or permission error.
4. **Run the app:** `streamlit run app.py` → expand **“Load a sample from Kaggle …”** → **Download / refresh dataset** → pick a row → **Insert into text area below**.

The first download is cached under Kaggle’s cache directory; no extra database server is required.