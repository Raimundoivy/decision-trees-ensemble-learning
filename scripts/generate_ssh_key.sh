#!/usr/bin/env bash
set -euo pipefail

# Generate an SSH key non-interactively to avoid the
# "Enter file in which to save the key" prompt from ssh-keygen.
# Usage: ./scripts/generate_ssh_key.sh [key_file_path]
# Example: ./scripts/generate_ssh_key.sh idk

KEY_FILE="${1:-idk}"

if [[ -z "${KEY_FILE}" ]]; then
  echo "Usage: $0 [key_file_path]" >&2
  exit 1
fi

if [[ -f "${KEY_FILE}" || -f "${KEY_FILE}.pub" ]]; then
  echo "Key file ${KEY_FILE} (or .pub) already exists. Skipping generation."
  exit 0
fi

ssh-keygen -t ed25519 -f "${KEY_FILE}" -N "" -q

echo "SSH key pair generated: ${KEY_FILE} and ${KEY_FILE}.pub"