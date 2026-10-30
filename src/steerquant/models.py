"""Explicit single-device Transformers backend; quantized models are never recast."""

import gc
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .io import digest, file_hash


def local_revision(path):
    files = {p.name: file_hash(p) for p in sorted(Path(path).iterdir())
             if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"}}
    if not files:
        raise ValueError("No local checkpoint files found")
    return "local:" + digest(files)


def lock_config(config):
    from copy import deepcopy
    from huggingface_hub import HfApi
    locked = deepcopy(config)
    model = locked["model"]
    if Path(model["id"]).is_dir():
        model["id"] = str(Path(model["id"]).resolve())
        model["revision"] = local_revision(model["id"])
    else:
        model["revision"] = HfApi().model_info(model["id"], revision=model["revision"]).sha
    return locked


def blocks(model):
    # These supported families expose residual block outputs at model.layers.
    if model.config.model_type not in {"llama", "qwen2", "mistral"}:
        raise ValueError(f"Unsupported architecture: {model.config.model_type}; add an explicit adapter")
    return model.model.layers


class Backend:
    def __init__(self, config, precision):
        self.config = config
        self.precision = precision
        m = config["model"]
        if precision not in config["precisions"]:
            raise ValueError("Precision is not enabled in this experiment")
        local = Path(m["id"]).is_dir()
        if local:
            if m["revision"] != local_revision(m["id"]):
                raise ValueError("Local checkpoint changed; lock the config again")
        elif not re.fullmatch(r"[0-9a-f]{40}", m["revision"]):
            raise ValueError("Pin model revision first with: steerquant lock")
        if m["device"].startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use the CPU smoke test locally")
        if precision in {"int8", "nf4"} and not m["device"].startswith("cuda"):
            raise ValueError("v1 quantization experiments require CUDA; CPU smoke uses fp32")
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[m["compute_dtype"]]
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp32":
            dtype = torch.float32
        common = {"trust_remote_code": False, "local_files_only": local}
        if not local:
            common["revision"] = m["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(m["id"], **common)
        if not self.tokenizer.chat_template:
            raise ValueError("An explicit chat template is required")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        kwargs = dict(common, dtype=dtype, attn_implementation=m["attention"])
        quantization = None
        if precision == "int8":
            quantization = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0,
                                            llm_int8_enable_fp32_cpu_offload=False)
        elif precision == "nf4":
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=False)
        if quantization is not None:
            kwargs.update(quantization_config=quantization, device_map={"": m["device"]})
        self.model = AutoModelForCausalLM.from_pretrained(m["id"], **kwargs)
        if quantization is None:
            self.model.to(m["device"])
        self.model.eval()
        self.layers = blocks(self.model)
        if any(l >= len(self.layers) for l in config["steering"]["layers"]):
            raise ValueError(f"Layer index exceeds model depth ({len(self.layers)})")
        from collections import Counter
        self.identity = {"model_id": m["id"], "revision": m["revision"], "precision": precision,
                         "model_type": self.model.config.model_type,
                         "hidden_size": self.model.config.hidden_size,
                         "chat_template_sha256": digest(self.tokenizer.chat_template)}
        self.runtime = {**self.identity, "device": m["device"], "requested_dtype": str(dtype),
                        "attention": m["attention"],
                        "quantization_config": quantization.to_dict() if quantization else None,
                        "parameter_dtypes": dict(Counter(str(p.dtype) for p in self.model.parameters())),
                        "linear_modules": dict(Counter(type(x).__name__ for x in self.model.modules()
                                                      if "Linear" in type(x).__name__)),
                        "memory_footprint_bytes": self.model.get_memory_footprint()}
        expected_quant_layer = {"int8": "Linear8bitLt", "nf4": "Linear4bit"}.get(precision)
        if expected_quant_layer and not self.runtime["linear_modules"].get(expected_quant_layer):
            raise ValueError(f"Requested {precision}, but no {expected_quant_layer} modules were loaded")
        if m["device"].startswith("cuda"):
            self.runtime["gpu"] = torch.cuda.get_device_name(torch.device(m["device"]))
            self.runtime["cuda_version"] = torch.version.cuda

    def render(self, text):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                 tokenize=False, add_generation_prompt=True)

    def encode(self, text, answer=""):
        result = self.tokenizer(self.render(text) + answer, add_special_tokens=False,
                                return_token_type_ids=False, return_tensors="pt")
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if max_positions and result["input_ids"].shape[1] > max_positions:
            raise ValueError("Prompt exceeds model context; silent truncation is disabled")
        return result.to(self.model.get_input_embeddings().weight.device)

    def generate(self, text):
        inputs = self.encode(text)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False,
                                         max_new_tokens=self.config["evaluation"]["max_new_tokens"],
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
        new = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip(), new.tolist()

    def close(self):
        del self.model
        self.layers = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
