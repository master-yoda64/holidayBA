#!/bin/sh

export UID="$(id -u)"
export GID="$(id -g)"

docker compose build --progress=plain --no-cache
docker compose up -d --remove-orphans
