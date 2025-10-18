# ---- Base image ----
FROM python:3.8-slim

# ---- Set working directory ----
WORKDIR /app

# ---- Copy all project files ----
COPY . /app

# ---- Install dependencies ----
RUN pip install --no-cache-dir -r requirements.txt

# ---- Expose Hugging Face default port ----
EXPOSE 7860

# ---- Run FastAPI using Uvicorn ----
CMD ["uvicorn", "call_api:app", "--host", "0.0.0.0", "--port", "7860"]
