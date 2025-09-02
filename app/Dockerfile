FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 1. Set working directory to a new root folder
WORKDIR /code

# 2. Copy requirements.txt to the new WORKDIR and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy your application code into a subdirectory named 'app'
COPY . /code/app

# Change ownership
RUN chown -R appuser:appuser /code

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8182

# 4. Run the application using the full module path, which now works
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8182"]