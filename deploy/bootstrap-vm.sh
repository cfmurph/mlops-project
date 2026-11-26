#!/usr/bin/env bash
# One-time setup of the PROD VM (Ubuntu 22.04/24.04, e.g. Oracle Cloud Always Free ARM).
# Run on the VM as the deploy user:  bash bootstrap-vm.sh
set -euo pipefail

# Docker Engine + compose plugin (official convenience script).
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"

# Where CI copies the compose files and writes envs/prod.env.
sudo mkdir -p /opt/housing/envs /opt/housing/logs /opt/housing/config
sudo chown -R "$USER" /opt/housing
# Airflow runs as uid 50000 inside the containers and writes logs/config here.
sudo chown -R 50000:0 /opt/housing/logs /opt/housing/config

# Host firewall: SSH and the public forecast API. Airflow's UI stays on localhost (use an SSH tunnel).
# Oracle images ship iptables rules; also allow 8000 in the VCN security list in the Oracle console.
if command -v ufw >/dev/null; then
  sudo ufw allow 22/tcp && sudo ufw allow 8000/tcp && sudo ufw --force enable
else
  sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
  sudo sh -c 'command -v netfilter-persistent >/dev/null && netfilter-persistent save || true'
fi

echo "Done. Log out and back in so the docker group applies, then run a deploy from GitHub Actions."
