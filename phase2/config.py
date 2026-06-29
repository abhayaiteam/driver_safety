import os
from dataclasses import dataclass


@dataclass
class Config:
    VLM_MODEL:           str   = os.getenv("VLM_MODEL",             "llava:7b")
    VLM_WORKERS:         int   = int(os.getenv("VLM_WORKERS",       "3"))
    VLM_ALERT_THRESHOLD: float = float(os.getenv("VLM_ALERT_THRESHOLD", "0.75"))
    LOG_FILE:            str   = os.getenv("LOG_FILE", "logs/driver_safety.log")


cfg = Config()
