# 1. Build the image
docker build -t pharma-clinical-rag .

# 2. Run the container (passing the Gemini key)
docker run -p 8501:8501 -e GEMINI_API_KEY="your_api_key_here" pharma-clinical-rag
