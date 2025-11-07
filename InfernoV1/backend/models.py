from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime

RunState = Literal[
    "PENDING","SCHEDULED","PROVISIONING","RUNNING","COLLECTING","COMPLETED","FAILED","ABORTED"
]

class Sweep(BaseModel):
    batch_size: Optional[List[int]] = None
    tensor_parallel: Optional[List[int]] = None
    
    # Quantization config
    dtype: Optional[List[str]] = None          # ["float16", "bfloat16", "float32"]
    quantization: Optional[List[str]] = None   # ["fp8", "fp4", "int8", "int4_awq", "none"]
    
    # TRT-LLM specific
    lookahead: Optional[List[int]] = None
    max_new_tokens: Optional[List[int]] = None
    input_tokens: Optional[List[int]] = None
    max_seq_len: Optional[List[int]] = None
    temperature: Optional[List[float]] = None
    top_p: Optional[List[float]] = None
    
    # Extra nested parameters
    extra: Dict[str, List[Any]] = Field(default_factory=dict)

class JobSpec(BaseModel):
    job_name: str
    model: str
    gpu_pool: Literal["A100-80GB","H100","H200","B200"]
    engine: Literal["vllm","trtllm"] = "vllm"
    dataset: Optional[str] = None
    base_env: Dict[str, str] = Field(default_factory=dict)
    sweep: Sweep
    strategy: Literal["grid","random"] = "grid"
    random_trials: Optional[int] = None
    num_gpus: int = 1
    timeout_s: int = 1800

class Run(BaseModel):
    run_id: str
    job_name: str
    spec: JobSpec
    config: Dict[str, Any]
    state: RunState
    created_at: datetime
    updated_at: datetime
    modal_job_id: Optional[str] = None
    paths: Dict[str, str] = Field(default_factory=dict)

class Metrics(BaseModel):
    model: str
    gpu: str
    config: Dict[str, Any]
    throughput_tok_s: float
    ttft_s: Optional[float] = None
    accuracy: Optional[float] = None
    timestamp: datetime