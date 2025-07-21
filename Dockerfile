# Blood_Vessel_Segmentation_UNET/Dockerfile
FROM python:3.12-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy the rest of your application
COPY . /app

# Set the working directory
WORKDIR /app

# Install dependencies using uv
RUN uv sync --frozen

# Expose port for Streamlit
EXPOSE 8506

# Run the app using uv
CMD ["uv", "run", "streamlit", "run", "Vessel_Segmentation.py", "--server.port=8506", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]