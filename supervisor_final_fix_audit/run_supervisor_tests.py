"""After unzip: python -B run_supervisor_tests.py. No PYTHONPATH setup needed."""
from pathlib import Path
import subprocess,sys

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    raise SystemExit(subprocess.call([sys.executable,'-B','-m','pytest',str(root/'foundation_successor_models/tests'),'-q',*sys.argv[1:]],cwd=root))
