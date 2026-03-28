# Download the Python Image
FROM python:3.12-slim

# Copy the entire project
COPY . credit_default_pipeline/

# Install the package with dependencies
RUN pip install ./credit_default_pipeline

# Set the Working Directory
WORKDIR /credit_default_pipeline

# Expose the port gunicorn will listen on
EXPOSE 5001

#Run gunicorn
CMD ["gunicorn", "--bind=0.0.0.0:5001", "app.main:app"]