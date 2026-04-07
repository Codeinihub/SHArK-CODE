#!/usr/bin/env python3
"""Quick test of the model improvements"""
import subprocess
import sys

result = subprocess.run([sys.executable, "FeatExtract.py"], 
                       capture_output=True, text=True,
                       cwd=r"c:\Users\User\OneDrive\Dokumenty\SHArK CODE")

# Get last 120 lines (where model results are)
lines = result.stdout.split('\n')
output = '\n'.join(lines[-120:])
print(output)

if result.returncode != 0:
    print("\nSTDERR:")
    print(result.stderr)
