# Use the official Python image from the Docker Hub

FROM python:3.9-slim


# Set the working directory

WORKDIR /app


# Install system dependencies

RUN apt-get update && \

    apt-get install -y --no-install-recommends apt-utils && \

    apt-get install -y curl && \

    apt-get install -y libgomp1 && \

    apt-get install -y gcc g++ && \

    ln -s /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/libgomp.so.1 && \

    apt-get clean && \

    rm -rf /var/lib/apt/lists/*


# Copy the requirements file

COPY requirements.txt requirements.txt


# Install the dependencies

RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application

COPY . .

# Ensure the .streamlit directory and config.toml are copied
COPY .streamlit /root/.streamlit

# Expose the port Streamlit runs on

EXPOSE 8501


# Run the Streamlit app

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

