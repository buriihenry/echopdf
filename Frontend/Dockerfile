FROM python:3.12-slim
EXPOSE 8084
WORKDIR /app

# Copy requirements file
COPY requirements.txt ./

# Install dependencies explicitly
RUN pip install --no-cache-dir streamlit boto3 langchain langchain-community faiss-cpu

# Copy the rest of your application
COPY . ./

# Use the Python executable to run streamlit
ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", "--server.port=8084", "--server.address=0.0.0.0"]