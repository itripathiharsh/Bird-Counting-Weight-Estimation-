# Bird Counting and Weight Estimation System

## Overview
This prototype analyzes CCTV footage of poultry. It performs:
1.  **Bird Counting:** Uses YOLOv8 + ByteTrack to detect and track birds, handling ID switches and occlusions.
2.  **Weight Estimation:** Calculates a "Weight Proxy" based on the pixel area of the bird's bounding box ($W \times H$).
3.  **API:** A FastAPI interface to upload video and receive analytics.

## Setup Instructions

### Prerequisites
* Python 3.9+
* CUDA capable GPU recommended (but works on CPU)

### Installation
1.  Extract the zip file.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the API
Start the server:
```bash
python main.py