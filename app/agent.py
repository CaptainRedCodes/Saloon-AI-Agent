from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    JobContext,
    RunContext,
)
from livekit.agents.llm import function_tool
from datetime import datetime, timezone
import logging

from app.knowledge_base import KnowledgeManager
from app.booking_manager import BookingManager
from app.help_request import HelpRequestManager
from app.models.booking import  BookingCreate, BookingUpdate 
from app.models.help_request import HelpRequestCreate
from app.models.salon_model import SalonUserData,AvailabilityCheckPayload
from app.information import INSTRUCTIONS

import asyncio

import json

with open("app/json/info.json","r") as f:
    app_data = json.load(f);
SALON_INFO = (
    app_data["name"],
    app_data["address"],
    app_data["contact"],
    app_data["working_hours"]
)
SALON_SERVICES =app_data["services"]


asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class Assistant(Agent):
    """Context-aware voice assistant for a hair salon."""
    
    def __init__(self, job_context: JobContext):
        self.job_context = job_context
        self.salon_info = SALON_INFO
        self.service_prices = SALON_SERVICES

        self.knowledge_base = KnowledgeManager()
        self.booking_manager = BookingManager()
        self.help_manager = HelpRequestManager()
        
        super().__init__(instructions=INSTRUCTIONS)
                    
        logger.info("Context-aware SalonAssistant initialized successfully")

    @function_tool
    async def get_current_date_and_time(self,context: RunContext[SalonUserData]) -> str:
        """
        Returns the current date, day of the week, and time in human-readable format for the AI agent.
        Updates the agent's context with last tool called and result.
        """
        # Get current UTC time and convert to local timezone
        now = datetime.now(timezone.utc).astimezone()
        
        day_name = now.strftime("%A")
        date_str = now.strftime("%B %d, %Y")
        time_str = now.strftime("%I:%M %p")
        
        human_readable = f"{day_name}, {date_str} at {time_str}"
        iso_format = now.isoformat()

        if getattr(context, "userdata", None):
            context.userdata.last_tool_called = "get_current_date_and_time"
            context.userdata.last_tool_result = {
                "day": day_name,
                "date": date_str,
                "time": time_str,
                "human_readable": human_readable,
                "iso": iso_format
            }

        return f"The current date and time is {human_readable}"
    @function_tool
    async def update_booking_context(
        self,
        context: RunContext[SalonUserData],
        request: BookingUpdate
    ) -> str:
        """
        Update the booking context with customer information using a payload model.
        """
        booking = context.userdata.current_booking
        updated_fields = []

        try:
            # Update name
            if request.customer_name:
                booking.customer_name = request.customer_name
                updated_fields.append("name")

            # Update phone number
            if request.phone_number:
                clean_phone = ''.join(filter(str.isdigit, request.phone_number))
                booking.phone_number = clean_phone
                updated_fields.append("phone")

            # Update service
            if request.service:
                service_lower = request.service.lower()
                if service_lower in self.service_prices:
                    booking.service = request.service
                    booking.price = self.service_prices[service_lower]
                    updated_fields.append("service")
                else:
                    available_services = ", ".join([s.title() for s in self.service_prices.keys()])
                    return f"'{request.service}' is not available. Our services are: {available_services}"

            # Update date
            if request.appointment_date:
                booking.appointment_date = request.appointment_date
                updated_fields.append("date")

            # Update time
            if request.appointment_time:
                booking.appointment_time = request.appointment_time
                updated_fields.append("time")

            # Update context state
            context.userdata.last_tool_called = "update_booking_context"
            context.userdata.last_tool_result = updated_fields

            # Check completeness
            if booking.is_complete():
                context.userdata.conversation_state = "ready_for_confirmation"
                return f"Great! I've updated: {', '.join(updated_fields)}. I now have all your information. Let me summarize everything for you."
            else:
                missing = [f for f in ["name","phone number","service","date","time"] 
                        if getattr(booking, f.replace(" ", "_")) in (None, "")]
                return f"I've updated: {', '.join(updated_fields)}. I still need: {', '.join(missing)}."

        except Exception as e:
            logger.error(f"Context update failed: {e}")
            return "I had trouble saving that information. Could you repeat it?"

    @function_tool
    async def get_booking_summary(self, context: RunContext[SalonUserData]) -> str:
        booking = context.userdata.current_booking
        context.userdata.last_tool_called = "get_booking_summary"

        if not booking.is_complete():
            missing = [f for f in ["name","phone number","service","date","time"] 
                    if getattr(booking, f.replace(" ", "_")) in (None, "")]
            return f"Booking is incomplete. Still need: {', '.join(missing)}"

        summary = (
            f"Here's what I have:\n"
            f"Name: {booking.customer_name}\n"
            f"Phone: {booking.phone_number}\n"
            f"Service: {booking.service} (${booking.price})\n"
            f"Date: {booking.appointment_date}\n"
            f"Time: {booking.appointment_time}\n\n"
            f"Does everything look correct?"
        )

        context.userdata.waiting_for_confirmation = True
        return summary


    @function_tool
    async def book_appointment(self, context: RunContext[SalonUserData]) -> str:
        booking = context.userdata.current_booking

        if not booking.is_complete():
            return "Cannot book - missing required information. Please provide all details first."
        
        if not context.userdata.waiting_for_confirmation:
            return "Please let me summarize the booking details for confirmation first."

        slot_available = await self.check_availability(
            booking.appointment_date, booking.appointment_time
        )
        if not slot_available:
            return f"Sorry, the slot {booking.appointment_time} on {booking.appointment_date} is fully booked. Please choose another time."

        if not booking.customer_name:
            raise ValueError("Error in customer name")
        
        if not booking.service:
            raise ValueError("Error in service")
        
        if not booking.appointment_date or not booking.appointment_time:
            raise ValueError("Error in Timing")
        
        if not booking.price:
            raise ValueError("Error in pricing")
        
        payload = BookingCreate(
            customer_name=booking.customer_name,
            service=booking.service,
            appointment_date=booking.appointment_date,
            appointment_time=booking.appointment_time,
            price=booking.price,
            phone_number=booking.phone_number,
        )
        booking_obj = await self.booking_manager.create_booking(payload)

        confirmation_number = booking_obj.confirmation_number

        # Update context
        context.userdata.conversation_state = "completed"
        context.userdata.last_tool_called = "book_appointment"
        context.userdata.last_tool_result = confirmation_number
        booking.confirmed = True

        result = (
            f"Perfect! Your appointment is confirmed for {booking.service} "
            f"on {booking.appointment_date} at {booking.appointment_time}. "
            f"Your confirmation number is {confirmation_number}. "
            f"We'll see you then!"
        )

        # Reset for next booking
        context.userdata.reset_booking()

        return result

    @function_tool
    async def check_availability(
        self,
        context: RunContext[SalonUserData],
        request: AvailabilityCheckPayload
    ) -> str:
        """
        Check slot availability for a given date and optionally time.
        Stores the check in context for reference.
        """
        try:
            MAX_BOOKINGS_PER_SLOT = 2
            all_slots = ["9:00 AM", "10:00 AM", "11:00 AM", "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM"]

            # Track this check in context
            context.userdata.availability_checks.append({
                "date": request.date,
                "time": request.time or "",
                "timestamp": datetime.now().isoformat()
            })
            context.userdata.last_tool_called = "check_availability"

            # Fetch all bookings for the given date
            bookings_ref = self.booking_manager.db.collection("appointments")\
                .where("appointment_date", "==", request.date)
            bookings = bookings_ref.stream()
            
            slot_counts: dict[str, int] = {slot: 0 for slot in all_slots}
            for doc in bookings:
                data = doc.to_dict()
                slot_time = data.get("appointment_time")
                if slot_time in slot_counts:
                    slot_counts[slot_time] += 1

            # Helper: find available slots
            available_slots = [slot for slot, count in slot_counts.items() if count < MAX_BOOKINGS_PER_SLOT]

            if request.time:
                # Check specific time
                if request.time not in all_slots:
                    return f"{request.time} is outside our business hours. Available times: {', '.join(all_slots)}"
                
                if slot_counts.get(request.time, 0) < MAX_BOOKINGS_PER_SLOT:
                    context.userdata.last_tool_result = "available"
                    return f"{request.time} on {request.date} is available."
                else:
                    context.userdata.last_tool_result = {"booked": request.time, "alternatives": available_slots}
                    if available_slots:
                        return f"{request.time} is fully booked. Available slots on {request.date}: {', '.join(available_slots)}"
                    else:
                        return f"All slots on {request.date} are fully booked."
            else:
                context.userdata.last_tool_result = available_slots
                if available_slots:
                    slots_formatted = "\n".join(f"• {t}" for t in available_slots)
                    return f"Available times on {request.date}:\n{slots_formatted}"
                else:
                    return f"Unfortunately, we're fully booked on {request.date}. Would you like to check another date?"

        except Exception as e:
            logger.error(f"Availability check failed: {e}")
            return "I'm having trouble checking availability. Let me get help from my supervisor."

    @function_tool
    async def request_help(
        self,
        context: RunContext[SalonUserData],
        request: HelpRequestCreate
    ) -> str:
        """
        Try to answer from knowledge base, otherwise send to supervisor.
        
        Args:
            request: Contains the question and room_name
        """
        question = request.question.strip()
        
        logger.info(f"Help requested: {question}")
        
        try:
            kb_result = self.knowledge_base.search(question, threshold=0.7)
            
            if kb_result:
                logger.info(f"✓ KB answered: {question[:50]}...")
                context.userdata.last_tool_called = "request_help"
                context.userdata.last_tool_result = "kb_found"
                return kb_result["answer"]
            
            logger.info(f"✗ KB couldn't answer, sending to supervisor: {question[:50]}...")
            
            payload = HelpRequestCreate(
                question=question,
                room_name=request.room_name
            )
            
            request_id = await self.help_manager.create_help_request(payload)
            
            context.userdata.last_tool_called = "request_help"
            context.userdata.last_tool_result = f"supervisor_notified:{request_id}"
            
            logger.info(f"✓ Help request created: {request_id}")
            
            return (
                "I've sent your question to my supervisor. "
                "They'll get back to you shortly with the answer. "
                "Is there anything else I can help you with in the meantime?"
            )
            
        except Exception as e:
            logger.error(f"Error in request_help: {e}")
            return (
                "I'm having trouble right now. "
                "Please hold on and I'll get someone to assist you."
            )