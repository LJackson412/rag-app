"""
- Verwendet rag/index.py und rag/retrieval.py für audit use-case 
- Input sind "reqs" zu auditierende Anforderungen und spezifische Audit-Prompts

"""

import asyncio
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from db_audit.audits.config import AuditConfig
from db_audit.audits.report import generate_audit_report
from db_audit.audits.schema import AuditResult, LLMAuditResult
from db_audit.audits.state import InputAuditState, OutputAuditState, OverallAuditState
from db_audit.rag.index import index_files
from db_audit.rag.retrieval import retrieve_docs


async def index(
    state: OverallAuditState, config: RunnableConfig
) -> dict[str, Any]:
    
    audit_config = AuditConfig.from_runnable_config(config)
    
    if not audit_config.skip_index:
    
        index_states = await index_files(
            collection_id=state.workitem_element_id,
            paths=state.paths,
            doc_ids=state.doc_ids,
            mode=audit_config.mode
        )
        return {"index_states": index_states}
    return {"index_states": []}

async def audit(
    state: OverallAuditState, config: RunnableConfig
) -> dict[str, Any]:
    
    audit_config = AuditConfig.from_runnable_config(config)
      
    retrieval_states = await retrieve_docs(
        collection_id=state.workitem_element_id,
        queries=state.requirements,
        number_of_docs_to_retrieve = audit_config.number_of_docs_to_retrieve,
        generate_answer_prompt=audit_config.audit_prompt,
        generate_answer_schema=audit_config.audit_schema
    )
  
    for rs in retrieval_states:
        llm_audit_res = cast(LLMAuditResult, rs.llm_answer)
        if rs.llm_answer is None:
            continue

        audit_res = AuditResult.from_llm(llm_audit_res)
        audit_res.attach_documents(rs.filtered_docs)

        rs.llm_answer = audit_res

    requirement_audits: list[tuple[str, AuditResult]] = []
    for requirement, retrieval_state in zip(state.requirements, retrieval_states):
        audit_result = retrieval_state.llm_answer
        if isinstance(audit_result, AuditResult):
            requirement_audits.append((requirement, audit_result))

    report_html = await asyncio.to_thread(
        generate_audit_report, state.workitem_element_id, requirement_audits
    )

    return {"retrieval_states": retrieval_states, "audit_report_html": report_html}



builder = StateGraph(
    state_schema=OverallAuditState,
    input_schema=InputAuditState,
    output_schema=OutputAuditState,
    context_schema=AuditConfig,
)

builder.add_node("index", index)
builder.add_node("audit", audit)

builder.add_edge(START, "index")
builder.add_edge("index", "audit")
builder.add_edge("audit", END)


graph = builder.compile()
graph.name = "Audit"



