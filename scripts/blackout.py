"""
blackout.py
===========
A stealth blackout screen tool.
Covers the entire monitor with pure black, hides the cursor, and ignores
all mouse movement and normal keys.

Only exits when your SECRET KEY is pressed.
Default secret key: F12 (or customizable below)
"""

import tkinter as tk
import sys

# ── Config ─────────────────────────────────────────────────────────────
# Change this to whatever secret key you want:
# Examples: "<F12>", "<F9>", "<Escape>", "<Control-Shift-B>"
SECRET_KEY = "<F12>"
# ───────────────────────────────────────────────────────────────────────

root = tk.Tk()

# Make it full screen, borderless, always on top
root.attributes("-fullscreen", True)
root.attributes("-topmost", True)
root.configure(background="black")

# Hide the mouse cursor
root.config(cursor="none")

def exit_blackout(event=None):
    root.destroy()
    sys.exit(0)

# Bind the secret key to exit
root.bind(SECRET_KEY, exit_blackout)

# Prevent closing with Alt+F4 or accidental clicks
root.protocol("WM_DELETE_WINDOW", lambda: None)

print(f"Blackout active. Press {SECRET_KEY.strip('<>')} to restore screen.")
root.mainloop()
