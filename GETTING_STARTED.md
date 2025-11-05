# Getting Started

## Quick Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd multimodal-data-preprocessing-assignment

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify structure (optional - already created)
python3 scripts/setup_directories.py
```

## Add Your Data

### Face Images (Everyone)

Add 3 photos to `data/images/your_name/`:

- `neutral.jpg`
- `smiling.jpg`
- `surprised.jpg`

### Voice Recordings (Everyone)

Add 2 recordings to `data/audio/your_name/`:

- `yes_approve.wav`
- `confirm_transaction.wav`

## Work on Your Task

### Create Your Branch

```bash
git checkout -b feature/your-task-name
```

### Task Assignments

- **Alice**: Data merge + product model → `src/data_processing/merge_data.py`, `src/models/product_recommender.py`
- **Yassin**: Image processing + face recognition → `src/data_processing/image_processing.py`, `src/models/face_recognition.py`
- **Brian**: Audio processing + voice verification → `src/data_processing/audio_processing.py`, `src/models/voice_verification.py`
- **Cedric**: Pipeline integration → `src/pipeline/authentication_pipeline.py`, `scripts/run_authentication_system.py`

### Push Your Work

```bash
git add .
git commit -m "Your changes description"
git push origin feature/your-task-name
```

## Project Structure Note

**Empty folders contain README.md files** - this ensures they get committed to Git. Check each folder's README to see what should go inside.

## Need Help?

- Check the main `README.md` for full details
- Each empty folder has a README explaining what goes there
- Python module files have docstrings showing what to implement
