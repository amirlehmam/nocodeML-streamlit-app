#!/bin/bash

# Get the IP address of the Streamlit container
STREAMLIT_IP=$(sudo docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' nocodeml_streamlit)

# Update the Nginx configuration with the new IP address
sed -i "s|proxy_pass http://.*:8501;|proxy_pass http://${STREAMLIT_IP}:8501;|g" ./nginx/conf.d/nocode-ml.com.conf

# Restart the Nginx container to apply the changes
sudo docker-compose restart nginx

