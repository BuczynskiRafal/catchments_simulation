FROM nvidia/cuda:11.0-base AS build

# Use python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /code

# Copy requirements.txt into the container
COPY requirements.txt requirements.txt

# Add this line to copy the 'static' folder into the container
COPY static static

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the application code
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose the port the app runs on
EXPOSE 8000

# Start the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
