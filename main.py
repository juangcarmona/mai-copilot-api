import os
import sys
from pathlib import Path
import argparse
import uvicorn

# Add the `src` folder to the Python path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "src"
sys.path.append(str(src_path))

# Argument parser for dynamic model and configuration
parser = argparse.ArgumentParser(description="MAI Copilot API")
parser.add_argument("--model", required=True, help="Default model for code completion")
parser.add_argument("--chat-model", help="Model for chat completions")
parser.add_argument("--device", default="cpu", help="Device to run models (cpu or cuda)")
parser.add_argument("--port", type=int, default=8000, help="Port for the API")
args = parser.parse_args()

# Set environment variables
os.environ["DEFAULT_GENERATOR"] = args.model
if args.chat_model:
    os.environ["CHAT_GENERATOR"] = args.chat_model

# Import the app after setting environment variables
from mai.api import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=True)
