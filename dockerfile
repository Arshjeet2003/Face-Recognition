# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip

# Copy requirements file (we'll create it next)
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy your app code
COPY . .

# Expose a port if needed (e.g., 5000 for Flask)
EXPOSE 5000

# Run your app (adjust if different)
CMD ["python", "app.py"]
