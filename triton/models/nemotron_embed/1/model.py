"""
Triton python-backend model: Nemotron 3.5 Embed 1B (BF16) text embeddings.

Per row: TEXT and optional INPUT_TYPE ("query" -> "query: " prefix, anything else -> "passage: ").
Output: EMBEDDING [batch, embedding_dim], truncated from 2048 and re-L2-normalized
(the model card requires re-normalization after truncation).
"""
import json

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from sentence_transformers import SentenceTransformer


def _decode(values):
    out = []
    for v in values.reshape(-1):
        out.append(v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v))
    return out


class TritonPythonModel:

    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        params = cfg.get("parameters", {})

        def param(key, default):
            return params.get(key, {}).get("string_value", default)

        self.model_id = param("model_id", "nvidia/Nemotron-3.5-Embed-1B-BF16")
        self.dim = int(param("embedding_dim", "1024"))
        self.encode_batch = int(param("encode_batch", "32"))

        device_id = args.get("model_instance_device_id", "0")
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        full_dim = self.model.get_sentence_embedding_dimension()
        if full_dim is not None and self.dim > full_dim:
            raise ValueError(f"embedding_dim {self.dim} > model dimension {full_dim}")

        pb_utils.Logger.log_info(
            f"nemotron_embed loaded {self.model_id} on {self.device}: "
            f"dim={self.dim} (of {full_dim}), prompts={self.model.prompts}"
        )

    def _encode(self, texts, prompt_name):
        emb = self.model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=self.encode_batch,
            convert_to_numpy=True,
        ).astype(np.float32)
        emb = emb[:, : self.dim]
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                texts = _decode(pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy())
                type_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TYPE")
                if type_tensor is None:
                    types = ["passage"] * len(texts)
                else:
                    types = [t.strip().lower() for t in _decode(type_tensor.as_numpy())]
                    if len(types) != len(texts):
                        raise ValueError(f"INPUT_TYPE rows ({len(types)}) != TEXT rows ({len(texts)})")

                out = np.zeros((len(texts), self.dim), dtype=np.float32)
                query_idx = [i for i, t in enumerate(types) if t.startswith("q")]
                passage_idx = [i for i, t in enumerate(types) if not t.startswith("q")]
                if query_idx:
                    out[query_idx] = self._encode([texts[i] for i in query_idx], "query")
                if passage_idx:
                    out[passage_idx] = self._encode([texts[i] for i in passage_idx], "document")

                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("EMBEDDING", out)]))
            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(f"nemotron_embed: {e}")))
        return responses

    def finalize(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
