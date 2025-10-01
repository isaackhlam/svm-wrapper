FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV WATCHFILES_FORCE_POLLING=true
CMD ["fastapi", "dev", "main.py", "--host", "0.0.0.0"]
