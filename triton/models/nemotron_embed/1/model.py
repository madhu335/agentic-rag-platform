"""
Triton python-backend model: Nemotron 3.5 Embed 1B (BF16) text embeddings.

Per row: TEXT and optional INPUT_TYPE ("q*" e.g. "query" -> "query" prompt, anything else or
missing -> "document" prompt). Output: EMBEDDING [batch, embedding_dim], truncated from 2048 and
re-L2-normalized (the model card requires re-normalization after truncation).

Batched execute() (task #17):
  The dynamic batcher hands us N requests at once (total rows <= max_batch_size 64). Instead of
  encoding each request separately, every row from every request is flattened, partitioned by
  INPUT_TYPE, and each partition goes through ONE encode() call. Results are scattered back by
  (request_idx, row_idx), so responses come back in the original order.

  Kafka mapping: requests = producers, INPUT_TYPE = partition key, (request_idx, row_idx) = offset
  that makes the scatter-back deterministic. A bad request fails alone; a failed encode() fails
  only the requests that had rows in that partition.

config.pbtxt parameters (string_value): model_id, embedding_dim, encode_batch.
  encode_batch is the GPU micro-batch inside encode(); set it to 64 (= max_batch_size) if you want
  a full dynamic batch to run as a single forward pass.
"""
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import triton_python_backend_utils as pb_utils
from sentence_transformers import SentenceTransformer

QUERY = "query"
DOCUMENT = "document"


def _decode(values):
    out = []
    for v in values.reshape(-1):
        out.append(v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v))
    return out


def _prompt_for(raw_type):
    # Same rule as the pre-batching model.py: anything starting with "q" is a query, else a passage.
    return QUERY if raw_type.strip().lower().startswith("q") else DOCUMENT


class TritonPythonModel:

    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        params = cfg.get("parameters", {})

        def param(key, default):
            return params.get(key, {}).get("string_value", default)

        self.model_id = param("model_id", "nvidia/Nemotron-3.5-Embed-1B-BF16")
        self.dim = int(param("embedding_dim", "2048"))
        self.encode_batch = int(param("encode_batch", "32"))

        device_id = args.get("model_instance_device_id", "0")
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        self.model.eval()

        # sentence-transformers 6.x renamed get_sentence_embedding_dimension -> get_embedding_dimension
        get_dim = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension
        full_dim = get_dim()
        if full_dim is not None and self.dim > full_dim:
            raise ValueError(f"embedding_dim {self.dim} > model dimension {full_dim}")

        prompts = getattr(self.model, "prompts", {}) or {}
        missing = [p for p in (QUERY, DOCUMENT) if p not in prompts]
        if missing:
            raise ValueError(f"model is missing prompt(s) {missing}; has {list(prompts)}")

        # Warm-up so the first real batch doesn't pay CUDA kernel/allocator startup cost.
        with torch.inference_mode():
            self._encode(["warmup"], QUERY)
            self._encode(["warmup"], DOCUMENT)

        pb_utils.Logger.log_info(
            f"nemotron_embed loaded {self.model_id} on {self.device}: "
            f"dim={self.dim} (of {full_dim}), encode_batch={self.encode_batch}, "
            f"prompts={prompts}, batched execute() enabled"
        )

    def _encode(self, texts, prompt_name):
        """One encode() for a whole partition -> float32 [len(texts), dim], unit-norm."""
        emb = self.model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=self.encode_batch,
            convert_to_tensor=True,
            normalize_embeddings=False,  # normalize after truncation, not before
            show_progress_bar=False,
        )
        emb = F.normalize(emb[:, : self.dim].float(), p=2, dim=1)  # eps-guarded for zero vectors
        return emb.cpu().numpy()

    def _parse(self, request):
        """[(text, prompt_name), ...] for every row in one request. Raises ValueError."""
        text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
        if text_tensor is None:
            raise ValueError("missing input TEXT")
        texts = _decode(text_tensor.as_numpy())
        if not texts:
            raise ValueError("TEXT is empty")

        type_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TYPE")
        if type_tensor is None:
            types = [DOCUMENT] * len(texts)
        else:
            raw = _decode(type_tensor.as_numpy())
            if len(raw) == 1 and len(texts) > 1:
                raw = raw * len(texts)  # one INPUT_TYPE for the whole request
            if len(raw) != len(texts):
                raise ValueError(f"INPUT_TYPE rows ({len(raw)}) != TEXT rows ({len(texts)})")
            types = [_prompt_for(t) for t in raw]
        return list(zip(texts, types))

    def execute(self, requests):
        n = len(requests)
        errors = [None] * n                    # per-request error message
        outputs = [None] * n                   # per-request float32 [rows, dim]
        partitions = defaultdict(list)         # prompt_name -> [(req_idx, row_idx, text)]

        # 1) Gather: flatten all rows, partition by INPUT_TYPE. Bad requests fail alone.
        for ri, request in enumerate(requests):
            try:
                rows = self._parse(request)
            except Exception as e:
                errors[ri] = f"nemotron_embed: {e}"
                continue
            outputs[ri] = np.zeros((len(rows), self.dim), dtype=np.float32)
            for row_idx, (text, prompt_name) in enumerate(rows):
                partitions[prompt_name].append((ri, row_idx, text))

        # 2) Encode: one encode() per partition, scatter back by (req_idx, row_idx).
        with torch.inference_mode():
            for prompt_name, items in partitions.items():
                try:
                    emb = self._encode([text for _, _, text in items], prompt_name)
                except Exception as e:
                    msg = f"nemotron_embed: encode failed ({prompt_name}, {len(items)} rows): {e}"
                    pb_utils.Logger.log_error(msg)
                    for ri, _, _ in items:
                        errors[ri] = errors[ri] or msg
                    continue
                for (ri, row_idx, _), vec in zip(items, emb):
                    outputs[ri][row_idx] = vec

        # 3) Respond in original request order.
        responses = []
        for ri in range(n):
            if errors[ri] is not None:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(errors[ri])))
            else:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("EMBEDDING", outputs[ri])]))

        if n > 1:
            sizes = {k: len(v) for k, v in partitions.items()}
            pb_utils.Logger.log_verbose(f"nemotron_embed execute: {n} requests, rows={sizes}")
        return responses

    def finalize(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
