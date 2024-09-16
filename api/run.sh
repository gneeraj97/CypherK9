#!/bin/bash

if [ -f /proc/sys/kernel/random/uuid ]; then
    export API_KEY="$(cat /proc/sys/kernel/random/uuid)-$(cat /proc/sys/kernel/random/uuid)"
else
    export API_KEY="$(uuidgen)-$(uuidgen)"
fi

if [ -z "${LITELLM_BASE_URL}" ]; then
  echo "LITELLM_BASE_URL is not set or is empty"
  exit
fi
if [ -z "${LITELLM_API_KEY}" ]; then
  echo "LITELLM_API_KEY is not set or is empty"
  exit
fi


if [ ! -f "/usr/local/bin/docker-compose" ]; then
    sudo curl -s -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi


docker-compose up -d

echo -e "\n\nCYPHER_K9_API_KEY: $API_KEY (also written to 'cypherk9_api_key'.txt)\n\n"

echo $API_KEY > cypherk9_api_key.txt
