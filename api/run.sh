#!/bin/bash

if [ -f /proc/sys/kernel/random/uuid ]; then
    API_KEY="$(cat /proc/sys/kernel/random/uuid)-$(cat /proc/sys/kernel/random/uuid)"
else
    API_KEY="$(uuidgen)-$(uuidgen)"
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


cat << EOF > docker-compose.yaml
services:
  cypherk9-api:
    build:
      context: .
      dockerfile: cypherk9.Dockerfile
    environment:
      LITELLM_BASE_URL: $LITELLM_BASE_URL
      LITELLM_API_KEY: $LITELLM_API_KEY
  nginx:
    image: nginx:latest
    ports:
      - 8000:8000
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - cypherk9-api
EOF

cat << EOF > nginx.conf
events { }

http {
  upstream cypherk9 {
    server cypherk9-api:8000;
  }

  server {
    listen 8000;

    location / {
      if (\$http_authorization != "Bearer $API_KEY") {
        return 401;
      }

      proxy_pass http://cypherk9;
    }
  }
}
EOF


docker-compose up -d

echo -e "\n\nCYPHER_K9_API_KEY: $API_KEY\n\n"

echo $API_KEY > cypherk9_api_key.txt
