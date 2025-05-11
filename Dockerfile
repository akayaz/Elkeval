# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes app.py, prompts.py, retriever_utils.py, and the tabs/ directory
COPY . .

# Make port 8501 available to the world outside this container
# This is the default port Streamlit runs on
EXPOSE 8501

# Define environment variable for Streamlit
# This tells Streamlit to run on all available network interfaces
# and disables running the browser on startup (useful for headless environments)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLE_CORS=false

# Run app.py when the container launches
# The .env file should be handled outside the Docker image build,
# e.g., by mounting it or passing environment variables at runtime.
CMD ["streamlit", "run", "app.py"]
