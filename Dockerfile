FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p models

EXPOSE 8000
EXPOSE 8501

# Start the validated API and Streamlit UI.
CMD uvicorn app.api_secure:app --host 0.0.0.0 --port 8000 & streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
