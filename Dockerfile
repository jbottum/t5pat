FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the code file into the container
COPY t5pat.py .

# Install the required dependencies
RUN pip install --upgrade pip
RUN pip install torch torchvision
RUN pip install transformers
RUN pip install llama_index
RUN pip install sentence_transformers

# Set the command to run when the container starts
CMD ["python", "t5pat.py"]

