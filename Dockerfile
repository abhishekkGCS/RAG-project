# 1. Use a lightweight, official Python 3.11 base image
FROM python:3.11-slim

# 2. Set environment variables to optimize Python for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# 5. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the entire project code into the container
COPY . .

# 7. Expose Streamlit's default port
EXPOSE 8501

# 8. Command to launch Streamlit accessible from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]