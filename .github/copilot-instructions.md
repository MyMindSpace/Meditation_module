# Copilot Instructions for Meditation Module

## Big Picture Architecture
- The project is a modular meditation recommendation system with a layered pipeline:
  - **Preprocessing Unit** (`preprocessing_unit/`): Scripts for audio, video, diagnosis, and user feedback preprocessing. Input: JSON, audio, video. Output: processed JSON, features.
  - **Encoders** (`Encoders/`): Encode processed data into vector embeddings (audio, vision, diagnosis, user profile). Output: JSON embeddings in `encoder_output/`.
  - **Core Engine** (`Core_engine/`): Orchestrates fusion, decision management, quality monitoring, posture detection, and recommendation. Consumes encoder outputs and produces final recommendations.
  - **Data Folders**: `preprocess_input/` (raw data), `preprocess_output/` (processed features), `encoder_output/` (embeddings).

## Data Flow
- Input data (audio, video, diagnosis, feedback) is placed in `preprocess_input/`.
- Preprocessing scripts output to `preprocess_output/`.
- Encoders read from `preprocess_output/` and write embeddings to `encoder_output/`.
- Core engine modules read embeddings and processed data to generate recommendations and session plans.

## Developer Workflows
- **Preprocessing**: Run scripts in `preprocessing_unit/` with default paths (input: `preprocess_input/`, output: `preprocess_output/`).
- **Encoding**: Run scripts in `Encoders/` with output to `encoder_output/`.
- **Training**: Use `Core_engine/meditation_xgboost_trainer.py` to train XGBoost models (requires `meditation.csv`).
- **Recommendation**: Use `Core_engine/meditation_recommender.py` to generate session plans. Supports Hugging Face LLM integration via `--hf-token` or `HUGGINGFACE_HUB_TOKEN`.
- **Debugging**: Most scripts print progress and output file paths. Check output JSONs for results.

## Project-Specific Patterns
- **Path Conventions**: All scripts default to using `preprocess_input/`, `preprocess_output/`, and `encoder_output/` for IO. Avoid hardcoding other paths.
- **Embeddings**: Encoders output structured JSON with keys for each embedding type (preferences, progress, behavior, feedback, combined).
- **Fusion**: Core engine fuses multiple model outputs (MSM, PDM, ARM, TTS) using weighted confidences.
- **LLM Integration**: `meditation_recommender.py` can use Hugging Face LLMs for script generation if token is provided.

## Integration Points & Dependencies
- **Python 3.11** required. Key libraries: `librosa`, `soundfile`, `opencv-python`, `transformers`, `xgboost`, `joblib`, `pandas`.
- **External Models**: Some modules download large model files (e.g., `pytorch_model.bin`). Ensure internet access or pre-download.
- **Hugging Face**: For LLM-based recommendations, set `HUGGINGFACE_HUB_TOKEN` or use `--hf-token`.

## Examples
- Run audio preprocessor:
  ```powershell
  python preprocessing_unit/audio_preprocessor.py --input preprocess_input/LibriSpeech --output preprocess_output/audio_features
  ```
- Encode user profiles:
  ```powershell
  python Encoders/user_profile_encoder.py --input preprocess_output/user_feedback_processed.json --output encoder_output/user_profiles_encoded.json
  ```
- Generate a meditation plan (with LLM):
  ```powershell
  python Core_engine/meditation_recommender.py --fused preprocess_output/fused_decision.json --catalog Core_engine/meditation.csv --out Core_engine/meditation_plan.json --use-llama --hf-token hf_xxx...
  ```

## Key Files & Directories
- `preprocessing_unit/` — Preprocessing scripts
- `Encoders/` — Encoder scripts
- `Core_engine/` — Core orchestration, fusion, decision, recommendation
- `preprocess_input/`, `preprocess_output/`, `encoder_output/` — Data and artifact folders
- `Core_engine/meditation.csv` — Meditation catalog for recommendations

---
_If any section is unclear or missing, please provide feedback for improvement._
