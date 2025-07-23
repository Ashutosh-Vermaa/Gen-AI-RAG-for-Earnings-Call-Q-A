FROM python:3.10-slim

# Avoid interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

# Ensure logs directory exists
RUN mkdir -p logs

RUN pip install --no-cache-dir -r requirements.txt

# Add after your pip install
RUN python -c "from langchain.document_loaders import PyPDFLoader; print('PyPDFLoader imported successfully')"
RUN python -c "import pypdf; print('pypdf imported successfully')"
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
