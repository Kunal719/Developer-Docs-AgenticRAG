from models.evidence import EvidenceItem

def dedupe_evidence(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    seen_chunk_ids = set()
    seen_content = set()
    deduped = []

    for item in evidence:
        doc = item.document

        chunk_id = doc.metadata.get("chunk_id")
        content = " ".join(doc.page_content.split())

        if chunk_id in seen_chunk_ids:
            continue

        if content in seen_content:
            continue

        seen_chunk_ids.add(chunk_id)
        seen_content.add(content)
        deduped.append(item)

    return deduped