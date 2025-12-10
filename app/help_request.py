import asyncio
from datetime import datetime
import logging
import os
from typing import Dict, List, Optional
from uuid import uuid4

import httpx
from app.config.settings import help_settings
from app.db import FirebaseManager
from app.models.help_request import (
    HelpRequestCreate,
    HelpRequestStatus,
    HelpRequestView,
    SupervisorResponse
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
        self.webhook_url = os.getenv("WEBHOOK_URL")
  
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


        # Save to Firebase
        def write_doc():
            doc_ref = self.db.collection(self.collection_name).document(request_id)
            doc_ref.set(doc_data.model_dump())
            return request_id

        await self._run_in_executor(write_doc)
        logger.info(f"Help request created: {request_id}  - {payload.question}")

        # Notify supervisor via webhook
        await self._notify_supervisor(request_id, doc_data)

        return request_id

    async def _notify_supervisor(self, request_id: str, help_request: HelpRequestView):
        """Send webhook notification to supervisor with request details."""
        if not self.webhook_url:
            logger.warning("Supervisor webhook URL not configured (WEBHOOK_URL env var)")
            return

        # Prepare webhook payload
        webhook_payload = {
            "event": "help_request_created",
            "request_id": request_id,
            "question": help_request.question,
            "room_name": help_request.room_name,
            "created_at": help_request.created_at.isoformat(),
            "status": help_request.status
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    self.webhook_url,
                    json=webhook_payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if resp.status_code == 200:
                    logger.info(f"✓ Supervisor notified for request {request_id}")
                else:
                    logger.warning(f"✗ Supervisor webhook returned {resp.status_code}: {resp.text}")
                    
        except httpx.TimeoutException:
            logger.error(f"✗ Webhook timeout for request {request_id}")
        except httpx.ConnectError:
            logger.error(f"✗ Cannot connect to webhook URL: {self.webhook_url}")
        except Exception as e:
            logger.error(f"✗ Error notifying supervisor: {e}")

    async def resolve_help_request(
        self,
        request_id: str,
        supervisor_response: SupervisorResponse
    ) -> Dict:
        """Resolve a help request (called by supervisor)."""

        def update_doc():
            doc_ref = self.db.collection(self.collection_name).document(request_id)
            doc = doc_ref.get()

            if not doc.exists:
                raise ValueError(f"Help request {request_id} not found")
            
            data = doc.to_dict()
            if not data:
                raise ValueError(f"Help request {request_id} data is empty")
            
            response_time = (datetime.now() - data["created_at"]).total_seconds()

            update_data = {
                "status": HelpRequestStatus.RESOLVED.value,
                "answer": supervisor_response.answer,
                "resolution_notes": supervisor_response.resolution_notes,
                "updated_at": datetime.now(),
                "response_time_seconds": response_time,
                "resolved_by": "supervisor",
                "resolved_at": datetime.now()
            }
            doc_ref.update(update_data)
            return data, response_time

        help_request, response_time = await self._run_in_executor(update_doc)
        logger.info(f"Help request {request_id} resolved in {response_time:.1f}s")

        return {
            "request_id": request_id,
            "status": "resolved",
            "response_time_seconds": response_time,
            "original_question": help_request["question"],
            "answer": supervisor_response.answer
        }

    async def get_pending_requests(self) -> List[HelpRequestView]:
        """Fetch all pending help requests."""

        def fetch_pending():
            query = self.db.collection(self.collection_name).where(
                "status", "==", HelpRequestStatus.PENDING.value
            ).order_by("created_at", direction="DESCENDING")
            
            requests = []
            for doc in query.stream():
                try:
                    requests.append(HelpRequestView(id=doc.id, **doc.to_dict()))
                except Exception as e:
                    logger.error(f"Error parsing request {doc.id}: {e}")
            return requests

        return await self._run_in_executor(fetch_pending)

    async def get_request_by_id(self, request_id: str) -> Optional[HelpRequestView]:
        """Fetch a specific help request by ID."""

        def fetch_doc():
            doc_ref = self.db.collection(self.collection_name).document(request_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return None
                
            data = doc.to_dict()
            if not data:
                return None
                
            return HelpRequestView(id=doc.id, **data)

        return await self._run_in_executor(fetch_doc)
    
    async def get_all_requests(self, limit: int = 50) -> List[HelpRequestView]:
        """Fetch recent help requests (for supervisor dashboard)."""

        def fetch_all():
            query = self.db.collection(self.collection_name)\
                .order_by("created_at", direction="DESCENDING")\
                .limit(limit)
            
            requests = []
            for doc in query.stream():
                try:
                    requests.append(HelpRequestView(id=doc.id, **doc.to_dict()))
                except Exception as e:
                    logger.error(f"Error parsing request {doc.id}: {e}")
            return requests

        return await self._run_in_executor(fetch_all)

