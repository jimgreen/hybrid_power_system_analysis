#!/usr/bin/env bash
set -euo pipefail

IMAGE="${HEAVYDB_IMAGE:-heavyai/core-os-cpu:latest}"
CONTAINER_NAME="${HEAVYDB_CONTAINER:-heavydb}"
DATA_DIR="${HEAVYDB_DATA_DIR:-/var/lib/heavyai}"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This installer expects Ubuntu or another apt-based Linux distribution." >&2
  exit 1
fi

echo "Installing Docker prerequisites..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

echo "Configuring Docker apt repository..."
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc >/dev/null
sudo chmod a+r /etc/apt/keyrings/docker.asc

. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" |
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "Preparing HeavyDB data directory and configuration..."
sudo mkdir -p "${DATA_DIR}"
sudo chown "${USER}:${USER}" "${DATA_DIR}"

cat >"${DATA_DIR}/heavy.conf" <<EOF
port = 6274
http-port = 6278
calcite-port = 6279
data = "${DATA_DIR}"
null-div-by-zero = true

[web]
port = 6273
frontend = "/opt/heavyai/frontend"
EOF

echo "Starting ${CONTAINER_NAME} with image ${IMAGE}..."
if sudo docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  sudo docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

sudo docker run -d \
  --name "${CONTAINER_NAME}" \
  -v "${DATA_DIR}:${DATA_DIR}" \
  -p 6273-6278:6273-6278 \
  "${IMAGE}"

echo "Waiting for HeavyDB container status..."
sudo docker ps --filter "name=${CONTAINER_NAME}" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'

cat <<'EOF'

HeavyDB/OmniSciDB is starting.

Local web UI:
  http://127.0.0.1:6273

Default login:
  user: admin
  password: HyperInteractive

SQL shell:
  sudo docker exec -it heavydb bin/heavysql

EOF
