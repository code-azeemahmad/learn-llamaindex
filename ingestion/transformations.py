from llama_index.core.schema import BaseNode, TransformComponent


class MetadataEnrichmentTransformer(TransformComponent):
    """Custom transformation to enrich and normalize node metadata during ingestion."""

    def __call__(
        self,
        nodes: list[BaseNode],
        **kwargs,
    ) -> list[BaseNode]:
        for node in nodes:
            # 1. Add static ingestion source tracking
            node.metadata["source"] = "local_documents"

            # 2. Derive normalized document name
            file_name = node.metadata.get("file_name", "")
            if file_name:
                node.metadata["document_name"] = str(file_name).strip().lower()

            # 3. Derive document category based on filename
            file_name_lower = str(file_name).lower()
            if "rag" in file_name_lower:
                category = "rag"
            elif "python" in file_name_lower:
                category = "python"
            else:
                category = "general"

            node.metadata["category"] = category

        return nodes