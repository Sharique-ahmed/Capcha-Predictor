# Use the official Python 3.11 image with the Bullseye variant
FROM python:3.11-bullseye

# Update system packages and install dependencies for Python libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to prevent Python from writing .pyc files
# and to ensure that stdout and stderr are sent to the terminal
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set a working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /app

# Expose the application port (adjust if necessary)
EXPOSE 5000

# Start the application
CMD ["python3", "main.py"]
