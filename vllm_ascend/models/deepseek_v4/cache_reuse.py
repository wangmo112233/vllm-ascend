# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 eligibility for the common layerwise cache planner."""

from vllm.config import VllmConfig

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    SpecReuseRules,
)


def get_spec_reuse_rules() -> SpecReuseRules:
    """The common planner selects by model type and excludes draft caches."""
    return {AscendMLAAttentionSpec: True, AscendSlidingWindowMLASpec: True}


def uses_push_producer(config: VllmConfig) -> bool:
    transfer = config.kv_transfer_config
    if transfer is None or not transfer.is_kv_producer:
        return False

    def is_push(name, extra):
        return name == "MooncakePushConnector" or (name == "MooncakeConnectorV2" and bool(extra.get("use_layerwise")))

    if transfer.kv_connector != "MultiConnector":
        return is_push(transfer.kv_connector, transfer.kv_connector_extra_config)
    return any(
        child.get("kv_role") in ("kv_producer", "kv_both")
        and is_push(child["kv_connector"], child.get("kv_connector_extra_config") or {})
        for child in transfer.kv_connector_extra_config.get("connectors", ())
    )


def uses_dspark_push(config: VllmConfig) -> bool:
    spec = getattr(config, "speculative_config", None)
    return getattr(spec, "method", None) == "dspark" and uses_push_producer(config)


def validate_rotating_config(config: VllmConfig) -> None:
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePushConnector
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
        get_layerwise_gva_config,
    )

    if get_layerwise_gva_config(config.kv_transfer_config) is None:
        raise ValueError("DeepSeek-V4 rotation requires a GVA store to restore shared pages.")
    push = uses_push_producer(config)
    if push and not callable(getattr(MooncakePushConnector, "wait_for_layer_reuse", None)):
        raise ValueError("Rotating DeepSeek-V4 with Mooncake Push requires shared-slot completion support (#21891).")
    spec = config.speculative_config
    if spec is not None and spec.method == "dspark" and not push:
        raise ValueError("Rotating DeepSeek-V4 DSpark requires a Push producer for draft context writes.")
