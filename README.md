# Pets Classifier API

Image classification service for 37 dog and cat breeds (Oxford-IIIT Pets).
ResNet18 with transfer learning (PyTorch), served with FastAPI and Docker.

## Results (test set, 3669 images)

| Metric | Value |
|---|---|
| Accuracy | 86.7% |
| Precision (macro) | 87.5% |
| Recall (macro) | 86.6% |
| F1 (macro) | 86.4% |

Training uses a separate validation split (20%) with early stopping.
The test set is evaluated once, using the best weights.

## Known limitations

Overall accuracy hides weak classes. Per-class analysis (`reports/per_class.csv`)
shows American Pit Bull Terrier recall is only 0.31, mostly confused with
Staffordshire Bull Terrier and American Bulldog. See `reports/top_confusions.txt`.

## Run locally

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision fastapi uvicorn python-multipart pillow numpy scikit-learn
python src/train.py
uvicorn src.app:app
```

Then open http://127.0.0.1:8000/docs

## Run with Docker

The model weights (`models/resnet18_pets.pt`) are not in the repo, so train first:

```
python src/train.py
docker build -t pets-api .
docker run --rm -p 8000:8000 pets-api
```

## API

- `GET /health` returns service status.
- `POST /predict` accepts a JPEG or PNG (max 5 MB) and returns the top 3 breeds with confidence.

## Project structure

- `src/train.py`: training, validation, early stopping, test metrics
- `src/analyze.py`: per-class recall and most common confusions
- `src/app.py`: FastAPI service
- `Dockerfile`: CPU container for the API