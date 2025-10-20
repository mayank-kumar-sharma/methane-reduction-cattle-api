# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# (Optional) Install basic system dependencies if needed in future
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching layers)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port (Render uses $PORT automatically)
EXPOSE 8000

# Start FastAPI app with uvicorn
# IMPORTANT: exec form so Docker handles signals properly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
