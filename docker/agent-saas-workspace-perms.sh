#!/command/with-contenv sh
# shellcheck shell=sh
# Prepare the Agent SaaS workspace volume after the image has applied any
# HERMES_UID/HERMES_GID remapping. Only the mount root is changed: existing
# tenant/workspace contents keep their current ownership.
set -eu

workspace="${1:-/workspace}"
if [ ! -e "$workspace" ]; then
    exit 0
fi
if [ -L "$workspace" ] || [ ! -d "$workspace" ]; then
    echo "[agent-saas] refusing unsafe workspace root: $workspace" >&2
    exit 1
fi

chown "$(id -u hermes):$(id -g hermes)" "$workspace"
chmod 0770 "$workspace"
