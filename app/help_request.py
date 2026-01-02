import asyncio
from datetime import datetime
import logging
from uuid import uuid4
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
from app.config.settings import help_settings
from app.db import FirebaseManager
from app.embeddings import get_encoder
from app.knowledge_base import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL
from app.models.help_request import (
    HelpRequestCreate,
    HelpRequestStatus,
    HelpRequestView,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HelpRequestManager:
    """Manages help requests with webhook notifications to supervisor."""
    
    def __init__(self):
        self.firebase = FirebaseManager()
        self.db = self.firebase.get_firestore_client()
        self.collection_name = help_settings.collection_name
        self.qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        self.qdrant_collection = QDRANT_COLLECTION
        self.encoder = get_encoder()
  
    async def _run_in_executor(self, func, *args):
        """Run synchronous Firebase operations in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)
    
    async def create_help_request(self, payload: HelpRequestCreate) -> str:
        """Create a new help request and notify supervisor via webhook."""
        request_id = str(uuid4())
        timestamp = datetime.now()

        doc_data = HelpRequestView(
                id=request_id,
                question=payload.question,
                answer=None,
                status=HelpRequestStatus.PENDING.value,
                room_name=payload.room_name,
                created_at=timestamp,
                updated_at=timestamp,
                resolution_notes=None,
                response_time_seconds=None,
                resolved_by=None,
                resolved_at=None
            )


        def write_doc():
            doc_ref = self.db.collection(self.collection_name).document(request_id)
            logger.info(f"Saved in DB: {request_id}")
            doc_ref.set(doc_data.model_dump(mode='json', exclude_none=True))
            return request_id

        await self._run_in_executor(write_doc)
        logger.info(f"Help request created: {request_id}  - {payload.question}")

        return request_id

    async def _store_in_qdrant(self, question: str, answer: str, request_id: str):
        """Store resolved question-answer pair in Qdrant vector database."""
        try:
            embedding = self.encoder.encode(question).tolist()
            
            point = PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "question": question,
                    "answer": answer,
                    "type": "supervisor_resolved",
                    "request_id": request_id,
                    "created_at": datetime.now().isoformat()
                }
            )
            
            self.qdrant.upsert(
                collection_name=self.qdrant_collection,
                points=[point]
            )
            logger.info(f"Stored Q&A in Qdrant for request {request_id}")
            
        except Exception as e:
            logger.error(f"Error storing in Qdrant: {e}")

    async def search_similar_resolved_questions(self, query: str, limit: int = 3, score_threshold: float = 0.7):
        """Search for similar resolved questions in Qdrant."""
        try:
            # Create embedding for query
            query_embedding = self.encoder.encode(query).tolist()
            
            # Search in Qdrant
            results = self.qdrant.search(
                collection_name=self.qdrant_collection,
                query_vector=query_embedding,
                limit=limit,
                threshold=score_threshold
            )
            
            similar_qas = []
            for result in results:
                similar_qas.append({
                    "question": result.payload["question"],
                    "answer": result.payload["answer"],
                    "similarity_score": result.score,
                    "request_id": result.payload.get("request_id")
                })
            
            return similar_qas
            
        except Exception as e:
            logger.error(f"✗ Error searching Qdrant: {e}")
            return []
