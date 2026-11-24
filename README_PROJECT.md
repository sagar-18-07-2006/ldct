# LDCT-Recon: deep-learning enhancement of low-dose CT

Contributors: Aryaman Singh, Sagar Maheshwari. Course: Digital Image Processing.

A neural network that removes noise from low-radiation CT slices while keeping anatomy intact, compared with classical
image-processing filters on real patient data (Mayo Clinic LDCT-and-Projection-data, TCIA).

## Main results (abdomen, Siemens, 15 patients, simulated 25% dose)

Residual U-Net under 5-fold patient-level cross-validation (every patient tested by a model that never saw them):

| Metric | Low-dose input | Model output | Patients improved |
| --- | --- | --- | --- |
| PSNR (dB) | 30.23 ± 2.77 | 34.35 ± 2.34 | 15 / 15 |
| SSIM | 0.866 ± 0.088 | 0.920 ± 0.060 | 15 / 15 |
| RMSE (HU) | 13.5 ± 6.3 | 8.2 ± 3.2 | 15 / 15 |

On a 2-patient test, the model also beat the best tuned classical filter (non-local means) by about 2.2 dB (33.36 vs 31.16 dB).

**Scope:** abdominal CT only. The Mayo collection also contains chest and head scans; they are outside the scope of this project.

## Files

| File | Role |
| --- | --- |
| `dataset.py` | Organ settings, DICOM to HU conversion, patient splits, patch dataset |
| `model.py` | Five (plus one optional) architectures, loss, training, evaluation, `compare` |
| `extras.py` | Tuned classical baselines (Gaussian, median, non-local means) and explanation figures |
| `cv.py` | Patient-level k-fold cross-validation |
| `server.py`, `client.py`, `app_client.py` | Client-server architecture: inference server (FastAPI), client library and thin web client |
| `app_site.py` | The project website (reads real result files) |
| `app.py` | Local app for downloading, training and comparing |
| `test_serving.py` | End-to-end test of the server and client |

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
# install PyTorch with the exact command shown on pytorch.org for your CUDA version, then:
pip install numpy scipy scikit-image pydicom matplotlib pandas tqdm streamlit
pip install -r requirements-serving.txt
```

## Reproduce

```powershell
python dataset.py from-folder --organ liver --dicom-dir D:/ldct_dicom
python model.py compare --organ liver --epochs 30 --archs resunet unet dncnn redcnn cnn
python extras.py all --organ liver --arch resunet
python cv.py --organ liver --arch resunet --folds 5 --epochs 30
python test_serving.py
```

Run the website:
```powershell
uvicorn server:app --host 127.0.0.1 --port 8000     # terminal 1 (needed only for the Live demo page)
streamlit run app_site.py                           # terminal 2
```

Fixed seed 42. Hardware used: NVIDIA RTX 4060 Laptop GPU, about 6 to 8 minutes per 30-epoch model.

## Limitations

15 patients from one collection; one scanner vendor and one body region; simulated low dose; 5 mm slices; PSNR / SSIM / RMSE are not clinical
quality measures; no lesion-detection study; outputs are smoother than the routine-dose reference; loss weights were set by hand
and no ablation was run. **Research prototype, not a medical device, not for diagnosis.**

## Data and citation

McCollough, C. H. et al. *Low Dose CT Image and Projection Data (LDCT-and-Projection-data)*. The Cancer Imaging Archive.
https://doi.org/10.7937/9npb-2637 (CC BY 4.0). Clark, K. et al. The Cancer Imaging Archive (TCIA). *J Digit Imaging* 26(6), 2013.
The data are not included in this repository.