#!/bin/sh

export UID="$(id -u)"
export GID="$(id -g)"

xhost local:docker

# comment out your favorite shell
# docker compose exec -it elas-jazzy zsh
docker compose exec -it holiday_ba bash
