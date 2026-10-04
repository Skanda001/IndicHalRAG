"""
setup.py
========

Interactive setup helper. Creates .env file with your API keys.

Run ONCE before anything else:
    python setup.py
"""

import os

print("=" * 60)
print("HALRAG PROJECT SETUP")
print("=" * 60)
print()
print("This script will create your .env file.")
print("Get keys from:")
print("  Gemini  → https://aistudio.google.com/app/apikey")
print("  Sarvam  → https://dashboard.sarvam.ai")
print()

if os.path.exists(".env"):
    overwrite = input(".env already exists. Overwrite? [y/N]: ").strip().lower()
    if overwrite != "y":
        print("Aborted. Existing .env kept.")
        exit()

gemini_key = input("Enter GEMINI_API_KEY  : ").strip()
sarvam_key = input("Enter SARVAM_API_KEY  : ").strip()

with open(".env", "w") as f:
    f.write(f"GEMINI_API_KEY={gemini_key}\n")
    f.write(f"SARVAM_API_KEY={sarvam_key}\n")

print()
print("✅ .env file created.")
print()
print("Next steps:")
print("  1. pip install -r requirements.txt")
print("  2. python scripts/download_questions.py")
print("  3. python scripts/retrieve_passages.py")
print("  4. python scripts/generate_responses.py --limit 10   (test first!)")
print("  5. python scripts/generate_responses.py              (full run)")
