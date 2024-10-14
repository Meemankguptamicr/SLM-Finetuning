FROM mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:latest
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install flash-attn==2.6.3 --no-build-isolation