import logging
from typing import Any, Dict, cast
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import os
import json


from app.config.settings import knowledge_settings


QDRANT_COLLECTION = knowledge_settings.collection_name
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

with open("info.json","r") as f:
    FAQ = json.load(f);


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class KnowledgeManager:
    def __init__(self):
        self.collection_name = QDRANT_COLLECTION
        self.faq = FAQ
        
        self.qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
    def initialize(self):
        """Initialize Qdrant collection and sync FAQs."""
        self._init_collection()
        self._sync_faqs()
        
    def _init_collection(self):
        """Create collection if it doesn't exist."""
        try:
            self.qdrant.get_collection(self.collection_name)
            logger.info("Collection already exists")
        except Exception:
            embedding_size = self.encoder.get_sentence_embedding_dimension()
            if embedding_size is None:
                raise ValueError("Could not determine embedding dimension")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection '{self.collection_name}'")
    
    def _sync_faqs(self):
        """Sync FAQs to Qdrant."""
        points = []
        for faq in self.faq:
            vector = self.encoder.encode(faq["question"]).tolist()
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "question": faq["question"],
                    "answer": faq["answer"],
                    "category": "faq",
                    "source": "local"
                }
            )
            points.append(point)
        
        if points:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Synced {len(points)} FAQs to Qdrant")
    
    def search(self, query: str, threshold: float = 0.7, top_k: int = 3):
        """
        Simple semantic search.
        """
        query_vector = self.encoder.encode(query).tolist()
        
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        
        # Return best match if score is above threshold
        if results.points and results.points[0].score >= threshold:
            payload = cast(Dict[str, Any], results.points[0].payload or {})
            
            return {
                "answer": payload.get("answer"),
                "question": payload.get("question"),
                "score": results.points[0].score
            }
        
        return None
    
    def add_knowledge(self, question: str, answer: str, category: str = "general"):
        """Add new knowledge to the database."""
        vector = self.encoder.encode(question).tolist()
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "question": question,
                "answer": answer,
                "category": category,
                "source": "user_added"
            }
        )
        
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        logger.info(f"Added: {question[:50]}...")
    
    def close(self):
        """Close the Qdrant client."""
        self.qdrant.close()