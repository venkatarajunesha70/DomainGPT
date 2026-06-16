"""
One-time script to create the Pinecone index.
Run once before starting the API:
  python scripts/create_pinecone_index.py
"""
from pinecone import Pinecone, ServerlessSpec
from apps.api.core.config import get_settings

settings = get_settings()


def create_index():
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = [idx.name for idx in pc.list_indexes()]

    if settings.pinecone_index_name in existing:
        print(f"✅ Index '{settings.pinecone_index_name}' already exists.")
        return

    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=settings.pinecone_env),
    )
    print(f"✅ Pinecone index '{settings.pinecone_index_name}' created.")
    print(f"   Dimension: {settings.embedding_dimension} | Metric: cosine")


if __name__ == "__main__":
    create_index()
