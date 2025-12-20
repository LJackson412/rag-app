"""
orchestriert einen Audit Lauf (trigger → read → index → retrieval → write-back).
hier gehören auch Retry/Idempotenz/Checkpointing (oder delegiert an ein state_store).
"""