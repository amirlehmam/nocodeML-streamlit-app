#!/bin/bash

LOGFILE=~/NoCodeML/update_and_restart.log

echo "Starting update and restart script" >> $LOGFILE

# Navigate to the project directory

cd ~/NoCodeML

echo "Pulling latest changes from git repository" >> $LOGFILE

git pull origin main >> $LOGFILE 2>&1

echo "Rebuilding Docker containers" >> $LOGFILE

docker-compose build -d --no-cache >> $LOGFILE 2>&1

echo "Restarting Docker containers" >> $LOGFILE

docker-compose down >> $LOGFILE 2>&1

docker-compose up -d >> $LOGFILE 2>&1

echo "Update and restart complete" >> $LOGFILE

