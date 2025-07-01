FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
