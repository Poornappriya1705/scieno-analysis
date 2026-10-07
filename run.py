from pathlib import Path
import argparse
from src.scientometric.pipeline import run_pipeline

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate scientometric analysis from WoS and Scopus exports")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--input", default=None, help="Override input folder")
    p.add_argument("--output", default=None, help="Override output folder")
    args = p.parse_args()
    run_pipeline(Path(args.config), args.input, args.output)
