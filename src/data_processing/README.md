# Image Processing Module

Flexible and reusable image processing pipeline for facial recognition systems.

## Features

- **Auto-detection**: Automatically detects user directories from the images folder
- **Configurable**: Accept custom user lists or directories
- **VGG16 Embeddings**: Extract 512-dimensional features using pre-trained deep learning
- **Data Augmentation**: Apply 8 different transformations per image
- **Easy Integration**: Simple API for processing new datasets

## Usage

### Basic Usage (Auto-detect users)

```python
from src.data_processing.image_processing import ImageProcessor

# Automatically detect all users from data/images directory
processor = ImageProcessor()
features = processor.process_all_images()
processor.save_features_to_csv(features)
```

### Custom Users List

```python
# Process only specific users
processor = ImageProcessor(users=['Alice', 'Bob'])
features = processor.process_all_images()
```

### Custom Directory

```python
# Use a different images directory
processor = ImageProcessor(
    base_dir='/path/to/custom/images',
    users=['User1', 'User2', 'User3']
)
features = processor.process_all_images()
```

### Display Sample Images Only

```python
# Just display images without processing
processor = ImageProcessor(users=['Alice', 'Armstrong'])
processor.display_sample_images(save_output=True)
```

## Directory Structure

```
data/
  images/
    Alice/
      neutral.jpg
      smiling.jpg
      surprised.jpg
    Armstrong/
      neutral.jpg
      smiling.jpg
      surprised.jpg
    cedric/
      ...
    yassin/
      ...
```

## Configuration Options

### Initialize Parameters

- **base_dir** (str or Path, optional): Path to images directory. Auto-detects if None
- **target_size** (tuple): Image dimensions for VGG16 (default: (224, 224))
- **users** (list, optional): List of user names to process. Auto-detects if None

### Output Configuration

- **save_features_to_csv(df, output_path=None)**: 
  - If `output_path=None`, automatically saves to `{project_root}/data/features/image_features.csv`
  - Uses absolute paths via pathlib, so works from any execution context
  
- **display_sample_images(save_output=True)**:
  - If `save_output=True`, automatically saves to `{project_root}/reports/sample_images.png`
  - No fragile relative paths - works regardless of where script is run

### Augmentation Types

The processor applies these 8 augmentations per image:
1. Original (no changes)
2. Rotation (+15 degrees)
3. Horizontal flip
4. Brightness adjustment (+30%)
5. Contrast adjustment (1.2x)
6. Grayscale conversion
7. Gaussian blur
8. Zoom (1.1x)

## Output

Generates `image_features.csv` with:
- **Metadata**: user, expression, augmentation type
- **Features**: 512-dimensional VGG16 embeddings
- **Labels**: Numeric class identifiers

## Example: Adding New Team Members

```python
# Step 1: Add new user images to data/images/NewUser/
# Step 2: Run processor (will auto-detect new user)
processor = ImageProcessor()
features = processor.process_all_images()
processor.save_features_to_csv(features)

# Or explicitly specify the new user
processor = ImageProcessor(users=['Alice', 'Armstrong', 'cedric', 'yassin', 'NewUser'])
```

## Benefits of This Design

✅ **No hardcoding** - Users are configurable  
✅ **Auto-detection** - Discovers users automatically  
✅ **Reusable** - Works with any facial image dataset  
✅ **Flexible** - Easy to add/remove users  
✅ **Testable** - Can process subsets for testing
