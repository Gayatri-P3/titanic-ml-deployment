
# BASE IMAGE
FROM python:3.11-slim

# WORKING DIRECTORY
WORKDIR /app

# COPY FILES
COPY . /app

# INSTALL DEPENDENCIES
RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE PORT
EXPOSE 8080

# RUN FLASK APP
CMD ["python", "app.py"]