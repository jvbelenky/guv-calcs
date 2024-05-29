FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ies_utils ./ies_utils/
COPY update_ies.sh update_ies.sh
RUN cd ies_utils && make install

COPY . ./

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "guv_app.py", "--server.port=8501", "--server.address=0.0.0.0"]