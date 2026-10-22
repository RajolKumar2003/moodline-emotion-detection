from pathlib import Path

readme = """# Moodline — Emotion Detection using BiGRU

Moodline is a deep-learning based text emotion detection application that classifies a sentence into one of six emotion categories using a trained BiGRU model.

The project includes the complete training notebook, saved model artifacts, and a Streamlit web application for interactive predictions.

## Emotion Classes

The model predicts one of the following emotions:

- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

## Features

- Text-based emotion classification
- BiGRU deep-learning model
- Probability breakdown for all six emotion classes
- Confidence score for the predicted emotion
- Interactive Streamlit user interface
- Dynamic emotion-based UI accent
- Input validation up to 2000 characters
- Cached model loading for faster repeated predictions

## Text Preprocessing

The deployment uses the same preprocessing steps used with the trained model:

1. Convert text to lowercase
2. Remove apostrophes
3. Remove special characters and punctuation
4. Remove extra spaces
5. Convert text into sequences using the saved tokenizer
6. Pad sequences to a maximum length of 50

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Streamlit
- Pickle

## Project Structure

```text
NLP Project/
├── .streamlit/
│   └── config.toml
├── Artifacts/
│   ├── BiGRU_Model.keras
│   └── tokenizer.pkl
├── .gitignore
├── final_clean.ipynb
├── requirements.txt
├── streamlit_app.py
└── README.md
```

## Model Artifacts

The application loads:

```text
Artifacts/BiGRU_Model.keras
Artifacts/tokenizer.pkl
```

The trained model and tokenizer are loaded once and cached by Streamlit.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Create a virtual environment:

```bash
py -3.12 -m venv .venv
```

Activate it on Windows:

```bash
.\\.venv\\Scripts\\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Requirements

```text
streamlit>=1.40,<2
tensorflow-cpu==2.17.0
numpy==1.26.4
h5py==3.11.0
```

## Streamlit Deployment

To deploy on Streamlit Community Cloud:

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository.
4. Set the main file path to:

```text
streamlit_app.py
```

5. Select Python 3.12 in Advanced settings.
6. Deploy the application.

## Example

Input:

```text
I feel loved spending time with you.
```

Possible output:

```text
Predicted Emotion: Love
```

The application also displays the confidence score and the probability distribution across all six emotions.

## Notebook

`final_clean.ipynb` contains the model development workflow and is kept separate from the Streamlit deployment code.

## Author

Rajol Kumar  
M.Sc. Mathematics and Scientific Computing  
National Institute of Technology Warangal
"""

path = Path("/mnt/data/README.md")
path.write_text(readme, encoding="utf-8")
print(path)
