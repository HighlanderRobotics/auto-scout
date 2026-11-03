#!/bin/bash
cd "$(dirname "$0")" || exit 1
for candidate in python3 python3.15 python3.14 python3.13 python3.12 python3.11; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3,11))' 2>/dev/null; then
    "$candidate" run.py "$@"
    result=$?
    read -r -p "Press Enter to close… "
    exit "$result"
  fi
done
echo "Install Python 3.11 or newer from https://www.python.org/downloads/ then retry."
read -r -p "Press Enter to close… "
exit 1
