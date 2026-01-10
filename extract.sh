#!/bin/bash
# Extract git repository contents to a text file
# Linux/Mac wrapper for extract.py

python3 "$(dirname "$0")/extract.py" "$@"
