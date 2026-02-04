"""Auto-sync module for repository change detection and re-ingestion."""

from skill_retriever.sync.auto_heal import AutoHealer, FailureType
from skill_retriever.sync.config import SYNC_CONFIG, SyncConfig
from skill_retriever.sync.manager import SyncManager
from skill_retriever.sync.oss_scout import OSSScout
from skill_retriever.sync.pipeline import DiscoveryPipeline, run_pipeline
from skill_retriever.sync.poller import RepoPoller
from skill_retriever.sync.registry import RepoEntry, RepoRegistry
from skill_retriever.sync.webhook import WebhookServer

__all__ = [
    "AutoHealer",
    "DiscoveryPipeline",
    "FailureType",
    "OSSScout",
    "RepoEntry",
    "RepoPoller",
    "RepoRegistry",
    "SYNC_CONFIG",
    "SyncConfig",
    "SyncManager",
    "WebhookServer",
    "run_pipeline",
]
