FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and the generated model.pkl
COPY . .

EXPOSE 5000

# Start the Flask API
CMD ["python", "app.py"]