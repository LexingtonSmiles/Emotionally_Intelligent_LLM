# Training On Google Colab GPU

Use Colab for the full README training commands. The local machine is CPU-only, so full training is much slower here.

## Steps

1. Open Google Colab.
2. Upload `Emotion_Response_Colab.ipynb`, or upload the whole project folder to Google Drive.
3. In Colab, select `Runtime > Change runtime type > T4 GPU` or any available GPU.
4. Run the notebook cells from top to bottom.
5. Download the generated checkpoints:

```text
checkpoints/emotion.pt
checkpoints/emotion_metrics.json
checkpoints/generator.pt
checkpoints/input_generator.pt
```

6. Put those files back into this local project's `checkpoints/` folder.

Then local inference can use:

```powershell
python -m emotion_response.infer --text "I cannot believe you forgot my birthday." --classifier checkpoints/emotion.pt --generator checkpoints/generator.pt --input-generator checkpoints/input_generator.pt
```

## Notes

- The notebook clones or copies this project into Colab, installs dependencies, verifies GPU availability, trains the models, and zips the checkpoints.
- For long training, keep the project folder in Google Drive and set `PROJECT_DIR` to that Drive folder. Then checkpoints are written to Drive after every epoch.
- Training commands use `--resume`, so rerunning a cell continues from the epoch stored in the checkpoint instead of starting over.
- The classifier uses class-weighted loss, and the emotion-aware generator uses balanced emotion sampling to reduce neutral-label dominance.
- Full training quality depends on runtime length. A T4 GPU should be much faster than local CPU. The stronger default is 12 classifier epochs, 40 emotion-aware generator epochs, and 40 input-only generator epochs; reduce those if Colab runtime expires. Checkpoints are saved after every epoch.

## Fixing `FileNotFoundError: /Final Project`

That error means the notebook is pointing at a folder that does not exist in Colab. In Colab, paths usually start with `/content`, not `/`.

Use one of these:

```python
PROJECT_DIR = Path('/content/Final Project')
```

or, if the folder is in Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
PROJECT_DIR = Path('/content/drive/MyDrive/Final Project')
```

The project folder must contain both `requirements.txt` and the `emotion_response` directory.
