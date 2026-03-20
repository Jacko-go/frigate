#!/bin/bash
set -e

REPO_DIR="$HOME/frigate-pose"
CONTAINER_NAME="frigate"
IMAGE_NAME="frigate:latest"

echo "🔄 Pulling latest code..."
cd "$REPO_DIR"
git pull

echo "🔨 Building Docker image (this may take a while)..."
make local

echo "🛑 Stopping old container..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "🚀 Starting new container..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart=unless-stopped \
  --privileged \
  --shm-size=256mb \
  --tmpfs /tmp/cache:size=1G,mode=1777 \
  --network frigate_default \
  -p 5000:5000 \
  -p 8554:8554 \
  -v /etc/localtime:/etc/localtime:ro \
  -v /home/minime/frigate/config:/config \
  -v /mnt/frigate_nas/recordings:/media/frigate \
  -v ~/frigate-pose/models:/models \
  "$IMAGE_NAME"

echo "⏳ Waiting for Frigate to start..."
sleep 5

if docker ps | grep -q "$CONTAINER_NAME"; then
  echo "✅ Frigate is running!"
  echo "   UI: http://$(hostname -I | awk '{print $1}'):5000"
else
  echo "❌ Frigate failed to start. Check logs:"
  echo "   docker logs $CONTAINER_NAME"
fi
