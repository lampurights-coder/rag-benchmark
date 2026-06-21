import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

warnings.filterwarnings("ignore", message=".*prepare_for_model.*")
warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")

from milvus_rag.api.factory import create_app

app = create_app()
