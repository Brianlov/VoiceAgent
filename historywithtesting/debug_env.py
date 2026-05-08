
import sys
import os

print(f"✅ Python Executable: {sys.executable}")
print(f"✅ Current Working Directory: {os.getcwd()}")
print(f"✅ System Path:")
for p in sys.path:
    print(f"  - {p}")

print("\nAttempting to import google.generativeai...")
try:
    import google.generativeai as genai
    print(f"✅ Success! Version: {genai.__version__}")
    print(f"✅ Location: {genai.__file__}")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    # Try to import pip to see installed packages
    print("\nListing installed packages via pip...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "list"])
