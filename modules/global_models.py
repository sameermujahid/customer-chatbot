import threading
import logging
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from modules.config import LLM_MODEL_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Locks for thread safety
_jina_lock = threading.Lock()
_llm_lock = threading.Lock()

# Global model instances
_jina_model = None
_tokenizer = None
_llm_model = None

def get_jina_model():
    global _jina_model
    if _jina_model is None:
        with _jina_lock:
            if _jina_model is None:
                logger.info("Loading Jina Embeddings model (singleton)...")
                _jina_model = SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)
                try:
                    if torch.cuda.is_available():
                        _jina_model = _jina_model.to('cuda')
                    else:
                        _jina_model = _jina_model.to('cpu')
                except Exception as e:
                    logger.warning(f"Could not move Jina model to device: {e}")
    return _jina_model

def get_llm_tokenizer_and_model():
    global _tokenizer, _llm_model
    if _tokenizer is None or _llm_model is None:
        with _llm_lock:
            if _tokenizer is None or _llm_model is None:
                logger.info("Loading LLM tokenizer and model (singleton)...")
                _tokenizer = AutoTokenizer.from_pretrained(
                    LLM_MODEL_DIR,
                    trust_remote_code=True,
                    use_fast=True
                )
                _llm_model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL_DIR,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_cache=False,
                    load_in_4bit=True
                )
                try:
                    _llm_model.eval()
                except Exception as e:
                    logger.warning(f"Could not set LLM model to eval: {e}")
    return _tokenizer, _llm_model 