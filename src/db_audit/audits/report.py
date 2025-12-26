from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

from typing import Iterable

from langchain_core.documents import Document

from db_audit.audits.schema import AuditResult, SubRequirementAudit

BADGE_COLORS = {
    "vollstaendig_erfuellt": ("#0E9F6E", "#ECFDF3"),
    "teilweise_erfuellt": ("#F59E0B", "#FFFBEB"),
    "nicht_erfuellt": ("#DC2626", "#FEF2F2"),
    "nicht_beurteilbar": ("#6B7280", "#F3F4F6"),
}


def _judgment_badge(judgment: str) -> str:
    color, background = BADGE_COLORS.get(judgment, ("#2563EB", "#EFF6FF"))
    return (
        f"<span class='badge' style='color:{color}; background:{background}; border-color:{color};'>"
        f"{escape(judgment.replace('_', ' '))}</span>"
    )


def _document_summary(doc: Document) -> str:
    chunk_id = doc.metadata.get("id") or doc.metadata.get("chunk_id") or "n/a"
    title = doc.metadata.get("title") or "Unbenannter Ausschnitt"
    source = doc.metadata.get("source") or doc.metadata.get("document_id") or "Unbekannte Quelle"

    meta_items = "".join(
        f"<li><strong>{escape(str(k))}:</strong> {escape(str(v))}</li>" for k, v in doc.metadata.items()
    )

    return (
        "<div class='evidence-card'>"
        f"<div class='evidence-title'>{escape(title)}</div>"
        f"<div class='evidence-meta'>Quelle: {escape(source)} · Chunk: {escape(str(chunk_id))}</div>"
        f"<div class='meta-list'><ul>{meta_items}</ul></div>"
        f"<div class='content-preview'>{escape(doc.page_content[:400])}</div>"
        "</div>"
    )


def _render_sub_requirement(sub_requirement: SubRequirementAudit) -> str:
    evidence_blocks = "".join(_document_summary(doc) for doc in sub_requirement.evidence_docs)
    return (
        "<div class='sub-card'>"
        f"<div class='sub-header'>"
        f"<div class='sub-title'>{escape(sub_requirement.requirement)}</div>"
        f"{_judgment_badge(sub_requirement.judgment)}"
        "</div>"
        f"<p class='sub-statement'>{escape(sub_requirement.statement)}</p>"
        f"<div class='evidence-grid'>{evidence_blocks or '<div class=\'no-evidence\'>Keine Evidenz dokumentiert.</div>'}</div>"
        "</div>"
    )


def _render_requirement_card(requirement: str, audit_result: AuditResult, index: int) -> str:
    sub_sections = "".join(_render_sub_requirement(sr) for sr in audit_result.sub_requirements)
    return (
        "<section class='card'>"
        f"<div class='card-header'>"
        f"<div><div class='eyebrow'>Anforderung {index}</div>"
        f"<h2>{escape(requirement)}</h2></div>"
        f"{_judgment_badge(audit_result.overall_judgment)}"
        "</div>"
        f"<p class='overall-statement'>{escape(audit_result.overall_statement)}</p>"
        f"<div class='sub-grid'>{sub_sections}</div>"
        "</section>"
    )


def generate_audit_report(
    workitem_id: str,
    requirements_with_results: Iterable[tuple[str, AuditResult]],
    base_dir: str | Path = "Audits",
) -> str:
    """Render audit results into a reusable HTML string.

    Args:
        workitem_id: Identifier of the audited work item used for folder and file names.
        requirements_with_results: Iterable of ``(requirement, AuditResult)`` tuples.
            The iterable should already be filtered to contain only successfully
            generated audit results and must be ordered as they should appear in
            the report (typically matching the original requirements input).
        base_dir: Root folder in which the report directory will be created.

    The report is stored under ``Audits/<workitem_element_id>/<workitem_element_id>.html``
    and the rendered HTML string is returned for downstream use.
    """

    requirement_cards: list[str] = []
    for idx, (requirement_text, audit_result) in enumerate(
        requirements_with_results, start=1
    ):
        requirement_cards.append(
            _render_requirement_card(requirement_text, audit_result, idx)
        )

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    workitem_id = workitem_id or "unbekannt"

    html = f"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Audit Report – {escape(workitem_id)}</title>
  <style>
    :root {{
      --bg: #0b1221;
      --panel: #0f172a;
      --card: #111827;
      --muted: #9ca3af;
      --text: #e5e7eb;
      --accent: #2563eb;
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body {{
      margin: 0;
      padding: 32px;
      color: var(--text);
      background: radial-gradient(circle at 20% 20%, rgba(37,99,235,0.15), transparent 30%),
                  radial-gradient(circle at 80% 0%, rgba(16,185,129,0.12), transparent 25%),
                  var(--bg);
    }}
    .page {{
      max-width: 1100px;
      margin: 0 auto;
      background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(17,24,39,0.85));
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      padding: 28px;
      box-shadow: 0 30px 80px rgba(0,0,0,0.35);
      backdrop-filter: blur(8px);
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 28px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      padding-bottom: 16px;
    }}
    .title-block h1 {{
      margin: 0;
      font-size: 26px;
      letter-spacing: -0.01em;
    }}
    .muted {{ color: var(--muted); margin-top: 4px; }}
    .badge {{
      border: 1px solid;
      padding: 6px 10px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 12px;
      text-transform: capitalize;
    }}
    .card {{
      background: var(--card);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 14px;
      padding: 18px 20px;
      margin-bottom: 18px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.25);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }}
    .eyebrow {{
      text-transform: uppercase;
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.1em;
    }}
    h2 {{ margin: 4px 0 10px; font-size: 20px; }}
    .overall-statement {{ color: #cbd5e1; margin: 0 0 12px; line-height: 1.5; }}
    .sub-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .sub-card {{
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px;
      padding: 14px;
      background: rgba(255,255,255,0.02);
    }}
    .sub-header {{ display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; }}
    .sub-title {{ font-weight: 700; }}
    .sub-statement {{ color: #d1d5db; margin: 8px 0 12px; line-height: 1.5; }}
    .evidence-grid {{ display: grid; gap: 10px; grid-template-columns: 1fr; }}
    .evidence-card {{
      border: 1px dashed rgba(255,255,255,0.1);
      border-radius: 10px;
      padding: 10px 12px;
      background: rgba(37, 99, 235, 0.05);
    }}
    .evidence-title {{ font-weight: 700; }}
    .evidence-meta {{ color: var(--muted); font-size: 13px; margin: 4px 0 6px; }}
    .meta-list ul {{ padding-left: 18px; margin: 0 0 8px; color: #cbd5e1; }}
    .content-preview {{ background: rgba(255,255,255,0.03); padding: 8px; border-radius: 8px; color: #e5e7eb; font-size: 14px; }}
    .no-evidence {{ color: var(--muted); font-style: italic; }}
  </style>
</head>
<body>
  <div class="page">
    <header>
      <div class="title-block">
        <h1>Audit Report</h1>
        <div class="muted">Workitem: {escape(workitem_id)}</div>
        <div class="muted">Generiert am {escape(generated_at)}</div>
      </div>
      <div class="badge" style="border-color: var(--accent); color: var(--accent); background: rgba(37,99,235,0.1);">{len(requirement_cards)} Anforderungen</div>
    </header>
    {''.join(requirement_cards) or '<p class="muted">Keine Audit-Ergebnisse vorhanden.</p>'}
  </div>
</body>
</html>
    """

    output_dir = Path(base_dir) / workitem_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{workitem_id}.html"
    output_path.write_text(html, encoding="utf-8")

    return html