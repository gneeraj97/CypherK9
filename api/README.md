# Running

Requirements:
- [Docker](https://www.docker.com/)
- [LiteLLM configured](#litellm-configuration)

```
export LITELLM_BASE_URL="http://IP:8080/"
export LITELLM_API_KEY="sk-..."
bash run.sh
```

This will:
- Generate a random API key
- Install docker-compose (if it's not already installed)
- Build the correct `docker-compose.yml` file that builds cypherk9.Dockerfile
- Builds the correct `nginx.conf` file
- Kicks everything off with `docker-compose -d` which exposes port 8000
- Displays the API key and writes it to `cypherk9_api_key.txt`


# Calling the API

`$ curl -X POST "http://localhost:8000/generate_cypher/" -H "Authorization: Bearer <TOKEN>"  -H "Content-Type: application/json" -d '{"query": "Which users have non-null SPNs?"}'`


# LiteLLM Configuration

Install LiteLLM on an EC2 instance with the following- replace `RANDOM` with ~20 random characters from https://1password.com/password-generator/

```
sudo yum install docker git
sudo curl -s -L https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

sudo usermod -a -G docker ec2-user
id ec2-user
newgrp docker
sudo systemctl enable docker.service
sudo systemctl start docker.service

# might need to exit/re-login

git clone https://github.com/BerriAI/litellm
cd litellm
echo 'LITELLM_MASTER_KEY="sk-RANDOM"' > .env
echo 'LITELLM_SALT_KEY="sk-RANDOM"' >> .env
source .env

docker-compose up -d
```

Then configure a model with the following settings:
- **type:** openai-compatible
- **name:** deepseek-coder
- **litellm model:** openai//root/deepseek_cypher/
- **api_base:** http://IP:8080/v1/
- **api_key:** GUID-GUID

Also configure a model with the following settings:
- **type:** openai-compatible
- **name:** mistralai/Mistral-7B-Instruct-v0.2
- **litellm model:** togther_ai/mistralai/Mistral-7B-Instruct-v0.2
- **api_base:** https://api.together.xyz/
- **api_key:** TOGETHER_AI_KEY
