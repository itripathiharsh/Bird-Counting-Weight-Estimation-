# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# 1. Install system dependencies required for OpenCV
# (OpenCV needs these graphics libraries to run on Linux servers)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy your project files into the container
COPY . .

# 3. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 4. Create the artifacts folder and give it write permissions
# (Crucial because the app needs to save videos here)
RUN mkdir -p artifacts && chmod 777 artifacts

# 5. Hugging Face Spaces expects the app to run on port 7860
EXPOSE 7860

# 6. Command to start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]