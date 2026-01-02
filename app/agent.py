from typing import Optional
from dotenv import load_dotenv
from livekit.agents.llm import function_tool
from datetime import datetime, timezone
import logging

from app.knowledge_base import KnowledgeManager
from app.booking_manager import BookingManager
from app.help_request import HelpRequestManager
from app.models.booking import BookingCreate, BookingUpdate, CollectCustomerInformationArgs 
from app.models.help_request import HelpRequestCreate
from app.models.salon_model import SalonUserData
from app.information import INSTRUCTIONS

import asyncio
import json

from app.slot_booking import AvailabilityChecker

with open("app/json/info.json", "r") as f:
    app_data = json.load(f)

SALON_INFO = (
    app_data["name"],
    app_data["address"],
    app_data["contact"],
    app_data["working_hours"]
)
SALON_SERVICES = app_data["services"]

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Assistant:
    """Context-aware voice assistant for a hair salon."""
    
    def __init__(self, session: SalonUserData, ctx):
        """
        Initialize assistant with user-specific context.
        
        Args:
            session: User-specific data model
            ctx: JobContext from LiveKit
        """
        self._ctx = ctx
        self._userdata = session
        
        # Load salon info
        with open("app/json/info.json", "r") as f:
            app_data = json.load(f)
        
        self.salon_info = {
            "name": app_data["name"],
            "address": app_data["address"],
            "contact": app_data["contact"],
            "working_hours": app_data["working_hours"],
            "services": app_data["services"]
        }
        
        # Initialize managers
        self.availability_checker = AvailabilityChecker()
        self.knowledge_base = KnowledgeManager()
        self.booking_manager = BookingManager()
        self.help_manager = HelpRequestManager()
        
        logger.info("Assistant initialized successfully")
    
    @function_tool
    async def get_current_date_and_time(self) -> str:
        """
        Returns the current date, day of the week, and time in human-readable format.
        Use this when you need to know what day or time it is.
        
        Returns:
            str: Current date and time formatted for conversation
        """
        now = datetime.now(timezone.utc).astimezone()
        
        day_name = now.strftime("%A")
        date_str = now.strftime("%B %d, %Y")
        time_str = now.strftime("%I:%M %p")
        
        human_readable = f"{day_name}, {date_str} at {time_str}"
        iso_format = now.isoformat()
        
        # Update context
        self._userdata.last_tool_called = "get_current_date_and_time"
        self._userdata.last_tool_result = {
            "day": day_name,
            "date": date_str,
            "time": time_str,
            "human_readable": human_readable,
            "iso": iso_format,
        }
        
        return f"The current date and time is {human_readable}"
    
    @function_tool
    async def collect_customer_information(
    self,
    customer: CollectCustomerInformationArgs,
) -> str:
        """
        Collect and store customer's personal information (name and phone).
        Use this tool FIRST when starting a new booking.
        """

        booking = self._userdata.current_booking
        updated_fields = []
        errors = []

        customer_name = customer.customer_name
        phone_number = customer.phone_number

        try:
            # Update name
            if customer_name:
                booking.customer_name = customer_name.strip()
                updated_fields.append("name")
                logger.info(f"Stored customer name: {customer_name}")

            # Update phone
            if phone_number:
                clean = ''.join(filter(str.isdigit,phone_number))
                if len(clean) != 10:
                    return "I couldn't catch a valid 10-digit phone number. Please repeat it slowly."
                booking.phone_number = clean
                updated_fields.append("phone number")
                logger.info(f"Stored phone number: {clean}")

            self._userdata.last_tool_called = "collect_customer_information"
            self._userdata.last_tool_result = updated_fields

            if errors:
                return f"I had trouble with: {', '.join(errors)}."

            if updated_fields:
                response = f"Got it! I've saved your {', '.join(updated_fields)}."

                missing = []
                if not booking.customer_name:
                    missing.append("name")
                if not booking.phone_number:
                    missing.append("phone number")

                if missing:
                    response += f" I still need your {', '.join(missing)}."
                else:
                    response += " Now, what service would you like to book?"

                return response

            return "Please provide your name and phone number."

        except Exception as e:
            logger.error("Failed to collect customer info", exc_info=True)
            return "I had trouble saving that information. Could you repeat it?"
    
    @function_tool
    async def select_service(
        self,
        service: str,
    ) -> str:
        """
        Select a service from available salon services.
        Use this after collecting customer information.
        
        Args:
            service: Service name (e.g., "haircut", "coloring", "styling")
            
        Returns:
            str: Confirmation with price and next steps
        """
        booking = self._userdata.current_booking
        
        try:
            # Validate customer info exists
            if not booking.customer_name or not booking.phone_number:
                return "I need your name and phone number first before selecting a service."
            
            # Validate and set service
            service_lower = service.lower().strip()
            if service_lower in self.salon_info['services']:
                booking.service = service
                booking.price = self.salon_info['services'][service_lower]
                
                self._userdata.last_tool_called = "select_service"
                self._userdata.last_tool_result = service
                
                logger.info(f"Service selected: {service} at ₹{booking.price}")
                
                return (
                    f"Perfect! {service.title()} costs ₹{booking.price}. "
                    "Now, when would you like to schedule your appointment? "
                    "Please provide a date and time."
                )
            else:
                available = ", ".join([s.title() for s in self.salon_info['services'].keys()])
                return (
                    f"I'm sorry, we don't offer '{service}'. "
                    f"Our available services are: {available}. "
                    "Which one would you like?"
                )
        
        except Exception as e:
            logger.error(f"Service selection failed: {e}", exc_info=True)
            return "I had trouble with that service selection. Could you try again?"
    
    @function_tool
    async def schedule_appointment(
        self,
        appointment_date: str,
        appointment_time: str,
    ) -> str:
        """
        Schedule the appointment with a specific date and time.
        Use this after customer info and service are collected.
        
        Args:
            appointment_date: Date in YYYY-MM-DD format (e.g., "2025-01-15")
            appointment_time: Time in HH:MM 24-hour format (e.g., "14:30" for 2:30 PM)
            
        Returns:
            str: Confirmation or availability status
        """
        booking = self._userdata.current_booking
        
        try:
            # Validate prerequisites
            if not booking.customer_name or not booking.phone_number:
                return "I need your name and phone number first."
            
            if not booking.service:
                return "Please select a service before scheduling a time."
            
            # Check availability first
            try:
                slot_available = self.availability_checker.check_availability(
                    appointment_date,
                    appointment_time
                )
                
                if not slot_available:
                    return (
                        f"Sorry, {appointment_time} on {appointment_date} is not available. "
                        "Would you like to check available slots for that date?"
                    )
            except Exception as e:
                logger.error(f"Availability check failed: {e}")
                return "I'm having trouble checking availability. Let me try again."
            
            # Store the appointment details
            booking.appointment_date = appointment_date
            booking.appointment_time = appointment_time
            
            self._userdata.last_tool_called = "schedule_appointment"
            self._userdata.last_tool_result = {
                "date": appointment_date,
                "time": appointment_time
            }
            
            logger.info(f"Appointment scheduled: {appointment_date} at {appointment_time}")
            
            # Move to confirmation state
            self._userdata.conversation_state = "ready_for_confirmation"
            
            return (
                f"Great! I've scheduled your {booking.service} for {appointment_date} "
                f"at {appointment_time}. Let me summarize everything for confirmation."
            )
        
        except ValueError as e:
            logger.error(f"Date/time validation error: {e}")
            return f"That date or time doesn't look right. Please use format YYYY-MM-DD for date and HH:MM for time."
        except Exception as e:
            logger.error(f"Scheduling failed: {e}", exc_info=True)
            return "I had trouble scheduling that. Could you try again?"
    
    @function_tool
    async def check_availability(
        self,
        date: Optional[str] = None,
        time: Optional[str] = None
    ) -> str:
        """
        Check available time slots for a specific date.
        Use this when customer wants to see what times are available.
        
        Args:
            date: Date in YYYY-MM-DD format
            time: Optional specific time in HH:MM format to check
            
        Returns:
            str: Available slots or specific time availability
        """
        try:
            if not date:
                return "Please provide a date to check availability."
            
            if not time:
                return "What time would you prefer for your appointment?"

            # Log the check
            self._userdata.availability_checks.append({
                "date": date,
                "time": time or "",
                "timestamp": datetime.now().isoformat()
            })
            self._userdata.last_tool_called = "check_availability"
            
            # Check availability
            result = self.availability_checker.check_availability(date, time)
            
            # Store result
            self._userdata.last_tool_result = {
                "status": result.status,
                "date": date,
                "time": time,
                "available_slots": getattr(result, 'available_slots', [])
            }
            
            logger.info(f"Availability checked for {date} {time or 'all slots'}")
            
            return result.message
            
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return f"That date format doesn't look right. Please use YYYY-MM-DD format."
        except Exception as e:
            logger.error(f"Availability check failed: {e}", exc_info=True)
            return "I'm having trouble checking availability right now. Please try again."
    
    @function_tool
    async def get_booking_summary(self) -> str:
        """
        Get a complete summary of the current booking for customer confirmation.
        Use this after all booking details are collected and before final confirmation.
        
        Returns:
            str: Formatted booking summary with all details
        """
        booking = self._userdata.current_booking
        self._userdata.last_tool_called = "get_booking_summary"
        
        if not booking.is_complete():
            missing = []
            if not booking.customer_name:
                missing.append("name")
            if not booking.phone_number:
                missing.append("phone number")
            if not booking.service:
                missing.append("service")
            if not booking.appointment_date:
                missing.append("date")
            if not booking.appointment_time:
                missing.append("time")
            
            return f"The booking is incomplete. I still need: {', '.join(missing)}"
        
        summary = (
            f"Let me confirm your booking details:\n"
            f"• Name: {booking.customer_name}\n"
            f"• Phone: {booking.phone_number}\n"
            f"• Service: {booking.service} (₹{booking.price})\n"
            f"• Date: {booking.appointment_date}\n"
            f"• Time: {booking.appointment_time}\n\n"
            f"Is everything correct? Say 'yes' to confirm or tell me what needs to be changed."
        )
        
        self._userdata.waiting_for_confirmation = True
        logger.info("Booking summary generated, waiting for confirmation")
        
        return summary
    
    @function_tool
    async def confirm_booking(self) -> str:
        """
        Finalize and confirm the booking after customer approval.
        ONLY use this after getting explicit customer confirmation (e.g., "yes", "correct", "confirm").
        
        Returns:
            str: Final confirmation with booking number
        """
        booking = self._userdata.current_booking
        
        # Validate completeness
        if not booking.is_complete():
            return "Cannot confirm - booking information is incomplete. Let me know what's missing."
        
        if not self._userdata.waiting_for_confirmation:
            return "Please let me show you the booking summary first so you can review it."
        
        # Final availability check
        try:
            if not booking.appointment_date or not booking.appointment_time:
                return "Booking information is incomplete. Please provide date and time."
            
            slot_available = self.availability_checker.check_availability(
                booking.appointment_date,
                booking.appointment_time
            )
            
            if not slot_available:
                return (
                    f"I'm sorry, but {booking.appointment_time} on {booking.appointment_date} "
                    "just became unavailable. Let me help you find another time."
                )
        except Exception as e:
            logger.error(f"Final availability check failed: {e}")
            return "I'm having trouble confirming availability. Please try again."
        
        # Create booking in system
        try:
            payload = BookingCreate(
                customer_name=booking.customer_name,
                service=booking.service,
                appointment_date=booking.appointment_date,
                appointment_time=booking.appointment_time,
                price=int(booking.price) if booking.price else None,
                phone_number=booking.phone_number,
            )
            
            booking_obj = await self.booking_manager.create_booking(payload)
            confirmation_number = booking_obj.confirmation_number
            
            # Update context
            self._userdata.conversation_state = "completed"
            self._userdata.last_tool_called = "confirm_booking"
            self._userdata.last_tool_result = confirmation_number
            booking.confirmed = True
            
            logger.info(f"Booking confirmed: {confirmation_number}")
            
            result = (
                f"Perfect! Your {booking.service} appointment is confirmed!\n"
                f"Date: {booking.appointment_date}\n"
                f"Time: {booking.appointment_time}\n"
                f"Confirmation Number: {confirmation_number}\n\n"
                f"We look forward to seeing you, {booking.customer_name}! "
                f"If you need to make changes, please call us at {self.salon_info['contact']}."
            )
            
            # Reset for next booking
            self._userdata.reset_booking()
            
            return result
            
        except Exception as e:
            logger.error(f"Booking creation failed: {e}", exc_info=True)
            return (
                "I encountered an error while confirming your booking. "
                f"Please call us directly at {self.salon_info['contact']} to complete your booking."
            )
    
    @function_tool
    async def modify_booking_detail(
        self,
        field: str,
        new_value: str,
    ) -> str:
        """
        Modify a specific detail in the current booking before confirmation.
        Use this when customer wants to change something they already provided.
        
        Args:
            field: What to change ("name", "phone", "service", "date", or "time")
            new_value: The new value for that field
            
        Returns:
            str: Confirmation of the change
        """
        booking = self._userdata.current_booking
        field = field.lower().strip()
        
        try:
            if field in ["name", "customer_name"]:
                booking.customer_name = new_value.strip()
                return f"Updated your name to {new_value}. Anything else to change?"
            
            elif field in ["phone", "phone_number"]:
                clean_phone = ''.join(filter(str.isdigit, new_value))
                if len(clean_phone) >= 10:
                    booking.phone_number = clean_phone[-10:]
                    return f"Updated your phone number. Anything else?"
                else:
                    return "Please provide a valid 10-digit phone number."
            
            elif field == "service":
                return await self.select_service(new_value)
            
            elif field == "date":
                booking.appointment_date = new_value
                return f"Updated appointment date to {new_value}. Anything else?"
            
            elif field == "time":
                booking.appointment_time = new_value
                return f"Updated appointment time to {new_value}. Anything else?"
            
            else:
                return f"I can modify: name, phone, service, date, or time. Which would you like to change?"
        
        except Exception as e:
            logger.error(f"Modification failed: {e}", exc_info=True)
            return "I had trouble making that change. Could you try again?"
    
    @function_tool
    async def get_salon_information(
        self,
        info_type: str = "all"
    ) -> str:
        """
        Get information about the salon (services, hours, contact, location).
        
        Args:
            info_type: Type of info ("services", "hours", "contact", "location", "all")
            
        Returns:
            str: Requested salon information
        """
        info_type = info_type.lower().strip()
        
        try:
            if info_type == "services":
                services_list = [
                    f"• {service.title()}: ₹{price}"
                    for service, price in self.salon_info['services'].items()
                ]
                return f"Our services:\n" + "\n".join(services_list)
            
            elif info_type == "hours":
                return f"We're open {self.salon_info['working_hours']}"
            
            elif info_type == "contact":
                return f"You can reach us at {self.salon_info['contact']}"
            
            elif info_type == "location":
                return f"We're located at {self.salon_info['address']}"
            
            else:  # "all"
                services_list = ", ".join([s.title() for s in self.salon_info['services'].keys()])
                return (
                    f"{self.salon_info['name']}\n"
                    f"Location: {self.salon_info['address']}\n"
                    f"Phone: {self.salon_info['contact']}\n"
                    f"Hours: {self.salon_info['working_hours']}\n"
                    f"Services: {services_list}"
                )
        
        except Exception as e:
            logger.error(f"Error getting salon info: {e}", exc_info=True)
            return "I'm having trouble retrieving that information right now."
    
    @function_tool
    async def request_help(
        self,
        question: str,
    ) -> str:
        """
        Answer customer questions using knowledge base or escalate to supervisor.
        Use this for questions not related to booking process.
        
        Args:
            question: The customer's question
            
        Returns:
            str: Answer or confirmation of escalation
        """
        question = question.strip()
        logger.info(f"Help requested: {question[:50]}...")
        
        try:
            # Try knowledge base first
            kb_result = self.knowledge_base.search(question, threshold=0.7)
            
            if kb_result:
                logger.info("Answered from knowledge base")
                self._userdata.last_tool_called = "request_help"
                self._userdata.last_tool_result = "kb_found"
                return kb_result["answer"]
            
            # Escalate to supervisor
            logger.info("Escalating to supervisor")
            
            self._userdata.last_tool_called = "request_help"
            self._userdata.last_tool_result = "supervisor_notified"
            
            return (
                "That's a great question! I've notified my supervisor who can provide "
                "more detailed information. They'll reach out to you shortly. "
                "In the meantime, is there anything else I can help with?"
            )
            
        except Exception as e:
            logger.error(f"Error in request_help: {e}", exc_info=True)
            return (
                "I'm having trouble right now. "
                "Please call us directly at {self.salon_info['contact']} for assistance."
            )