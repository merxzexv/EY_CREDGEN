from flask import Flask, request, jsonify, send_from_directory, render_template, session, redirect, url_for
from flask_cors import CORS
from dotenv import load_dotenv
import os
import uuid
import time
from datetime import datetime
import threading
import json
from werkzeug.utils import secure_filename

# Import the core agents
from agents.master_agent import MasterAgent, IntentType
from agents.underwriting_agent import UnderwritingAgent
from agents.sales_agent import SalesAgent
from agents.fraud_agent import FraudAgent
from utils.pdf_generator import generate_sanction_letter as generate_sanction_pdf
from models.gemini_service import GeminiService
from models.openrouter_service import OpenRouterService

# Load environment variables
load_dotenv()

# --- 1. Initialization ---
app = Flask(__name__)
CORS(app)

# Basic config and paths (for admin system)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
os.makedirs(CSV_DIR, exist_ok=True)

UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

APPLICATIONS_CSV = os.path.join(CSV_DIR, "applications.csv")
CHAT_LOGS_CSV = os.path.join(CSV_DIR, "chat_logs.csv")
TUNING_CSV = os.path.join(CSV_DIR, "tuning_content.csv")
ADMINS_CSV = os.path.join(CSV_DIR, "admin_users.csv")

app.secret_key = os.environ.get("APP_SECRET_KEY", "change-this-secret-key")
app.config["UPLOAD_FOLDER"] = UPLOADS_DIR
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# Session management with expiration
user_sessions = {}
SESSION_TIMEOUT = 1800  # 30 minutes in seconds

# Initialize agents
master_agent = MasterAgent()
underwriting_agent = UnderwritingAgent()
sales_agent = SalesAgent()
fraud_agent = FraudAgent()
# Initialize Active LLM Service
llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
llm_service = None

if llm_provider == "openrouter":
    llm_service = OpenRouterService()
    print(f"Initialized OpenRouter Service (Provider: {llm_provider})")
else:
    llm_service = GeminiService()
    print(f"Initialized Gemini Service (Provider: {llm_provider})")

# Cleanup thread for expired sessions
def cleanup_sessions():
    """Periodically clean up expired sessions"""
    while True:
        time.sleep(300)  # Run every 5 minutes
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session_data in user_sessions.items():
            if current_time - session_data.get('last_activity', 0) > SESSION_TIMEOUT:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del user_sessions[session_id]
            print(f"Cleaned up expired session: {session_id}")

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
cleanup_thread.start()

# --- Utility Functions ---

def get_session_id(request):
    """Retrieve or create a session ID with validation."""
    session_id = request.headers.get('X-Session-ID')
    
    if not session_id:
        # Generate new session ID
        session_id = f"session_{uuid.uuid4().hex[:16]}"
    
    return session_id

def update_session_activity(session_id):
    """Update last activity time for session."""
    if session_id in user_sessions:
        user_sessions[session_id]['last_activity'] = time.time()

def initialize_user_session(session_id):
    """Initialize a new session for a user."""
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            'master_agent': MasterAgent(),  # Create new instance per session
            'last_activity': time.time(),
            'created_at': datetime.now().isoformat(),
            'interaction_count': 0,
            'chat_history': []
        }
    
    update_session_activity(session_id)
    return user_sessions[session_id]

def generate_sanction_letter_text(state: dict) -> dict:
    """Generate the final sanction letter with all details."""
    entities = state.get('entities', {})
    current_offer = state.get('current_offer', {})
    
    # Extract data with fallbacks
    loan_amount = entities.get('loan_amount', 0)
    interest_rate = current_offer.get('interest_rate', state.get('interest_rate', 'N/A'))
    
    if isinstance(interest_rate, (int, float)):
        rate_str = f"{interest_rate:.2f}%"
        emi = current_offer.get('monthly_emi', calculate_emi(loan_amount, interest_rate, entities.get('tenure', 60)))
    else:
        rate_str = str(interest_rate)
        emi = "N/A"
    
    name = entities.get('name', 'Applicant')
    tenure = entities.get('tenure', 60)
    date_today = datetime.now().strftime("%B %d, %Y")

    # Format EMI safely for display
    if isinstance(emi, (int, float)):
        emi_str = f"{emi:,.2f}"
    else:
        emi_str = str(emi)
    
    letter_content = f"""
CREDGEN FINANCIAL SERVICES
==========================
SANCTION LETTER

Date: {date_today}
Sanction Letter No: SL-{uuid.uuid4().hex[:8].upper()}

Dear {name},

We are pleased to inform you that your loan application has been APPROVED.

Loan Details:
-------------
• Sanctioned Amount: ₹{loan_amount:,}
• Interest Rate: {rate_str}
• Loan Tenure: {tenure} months
• Monthly EMI: ₹{emi_str}
• Processing Fee: ₹{max(1000, loan_amount * 0.01):,}
• Disbursement Date: Within 3 working days

Terms & Conditions:
-------------------
1. This sanction is valid for 30 days from the date of issue.
2. Final disbursement is subject to document verification.
3. Rate of interest is subject to change as per market conditions.
4. Penalty for late payment: 2% per month.

Please sign and return this letter to proceed with disbursement.

For CredGen Financial Services,
[Authorized Signatory]

This is a computer-generated letter and does not require a signature.
"""
    
    return {
        'content': letter_content,
        'metadata': {
            'sanction_id': f"SL-{uuid.uuid4().hex[:8].upper()}",
            'date': date_today,
            'applicant': name,
            'amount': loan_amount,
            'interest_rate': rate_str,
            'tenure': tenure
        }
    }

def calculate_emi(principal, rate, tenure_months):
    """Calculate EMI (simplified)."""
    if tenure_months <= 0:
        return 0
    monthly_rate = rate / 12 / 100
    emi = principal * monthly_rate * (1 + monthly_rate) ** tenure_months / ((1 + monthly_rate) ** tenure_months - 1)
    return round(emi, 2)


# --- Admin/CSV Utilities ---
def ensure_csv(file_path, headers):
    exists = os.path.exists(file_path)
    if not exists:
        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            import csv as _csv
            writer = _csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()


def append_csv(file_path, row, headers):
    import csv as _csv
    ensure_csv(file_path, headers)
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=headers)
        writer.writerow(row)


def read_csv(file_path):
    import csv as _csv
    if not os.path.exists(file_path):
        return []
    with open(file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        return list(reader)


def log_chat_event(session_id, event_type, payload):
    import csv as _csv
    headers = [
        "timestamp",
        "session_id",
        "event_type",
        "message_role",
        "message_text",
        "status",
        "details",
    ]
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "event_type": event_type,
        "message_role": payload.get("role"),
        "message_text": payload.get("text"),
        "status": payload.get("status"),
        "details": json.dumps(payload.get("details", {}), ensure_ascii=False),
    }
    append_csv(CHAT_LOGS_CSV, row, headers)


ALLOWED_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def seed_default_admin():
    headers = ["username", "password"]
    ensure_csv(ADMINS_CSV, headers)
    admins = read_csv(ADMINS_CSV)
    if not admins:
        append_csv(ADMINS_CSV, {"username": "admin", "password": "admin123"}, headers)


seed_default_admin()


def init_csv_files():
    """Ensure core CSVs exist with headers so they are visible upfront."""
    ensure_csv(APPLICATIONS_CSV, [
        'timestamp', 'session_id', 'full_name', 'phone', 'email',
        'city', 'loan_amount', 'status', 'rejection_reason', 'attached_files'
    ])
    ensure_csv(CHAT_LOGS_CSV, [
        'timestamp', 'session_id', 'event_type', 'message_role',
        'message_text', 'status', 'details'
    ])
    ensure_csv(TUNING_CSV, ['timestamp', 'admin', 'type', 'content', 'filename'])


# Create empty CSV files with headers at startup
init_csv_files()

# --- Frontend Routes ---

@app.route('/')
def index():
    """Serve the main frontend page."""
    return send_from_directory('frontend', 'index.html')

@app.route('/widget.html')
def widget():
    """Serve the widget page for 3rd party embedding."""
    return send_from_directory('frontend', 'widget.html')

@app.route('/frontend/<path:filename>')
def frontend_files(filename):
    """Serve static frontend files (CSS, JS, etc.)."""
    return send_from_directory('frontend', filename)

# --- API Endpoints ---

def get_bank_context():
    """Retrieve and concatenate all tuning context from CSV."""
    try:
        rows = read_csv(TUNING_CSV)
        # Extract content field, filter empty
        contents = [r.get('content', '').strip() for r in rows if r.get('content', '').strip()]
        if contents:
            return "\n\n".join(contents)
    except Exception as e:
        print(f"Error loading bank context: {e}")
    return ""

@app.route('/chat', methods=['POST'])
def chat():
    """
    Primary conversational endpoint.
    """
    try:
        session_id = get_session_id(request)

        # Support both JSON and multipart/form-data (for file uploads)
        user_input = ''
        if request.content_type and 'multipart/form-data' in request.content_type:
            user_input = (request.form.get('message') or '').strip()
            # Files are accepted but currently not processed; this avoids errors when files are sent.
        else:
            data = request.get_json(silent=True) or {}
            user_input = (data.get('message') or '').strip()

        if not user_input:
            return jsonify({
                'message': 'Please provide a message.',
                'error': 'empty_input'
            }), 400
        
        # Initialize or retrieve session
        # Log the user message
        log_chat_event(session_id, 'message', {
            'role': 'user',
            'text': user_input,
            'status': None,
            'details': {}
        })

        session = initialize_user_session(session_id)
        session['interaction_count'] += 1
        
        # Get the user's master agent instance
        user_master_agent = session['master_agent']
        
        # Process the input
        # --- MEDIATOR LOGIC ---
        gemini_mode = os.getenv("LLM_MODE", "disabled").lower().strip()
        
        # Load base prompt and dynamic context
        base_prompt = os.getenv("LLM_SYSTEM_PROMPT")
        bank_context = get_bank_context()
        
        if bank_context:
            system_prompt = f"{base_prompt}\n\n[BANK SPECIFIC CONTEXT]\n{bank_context}\n[END CONTEXT]"
        else:
            system_prompt = base_prompt

        response = None
        
        # Mode 1: Enabled (AI Only / Primary)
        # Mode 1: Enabled (AI Only / Primary)
        if gemini_mode == "enabled":
            try:
                # --- BACKEND LOGIC RESTORATION ---
                # 1. Update interaction history
                if 'chat_history' not in session:
                    session['chat_history'] = []
                
                # 2. Extract entities and update state (The "Storage" part)
                detected_entities = user_master_agent.extract_entities(user_input)
                detected_intent, _ = user_master_agent.detect_intent(user_input)
                
                # Only update if meaningful
                user_master_agent.update_state(detected_entities, detected_intent)
                
                # 3. Check for specific transition to workers based on collected data
                # This ensures we don't just "talk" but actually "do" things
                worker_to_trigger = "none"
                current_stage = user_master_agent.state.get("stage")
                
                # String comparison to be safe if Enum not imported
                stage_val = current_stage.value if hasattr(current_stage, "value") else str(current_stage)
                
                if stage_val == "underwriting":
                    worker_to_trigger = "underwriting"
                elif stage_val == "fraud_check":
                    worker_to_trigger = "fraud"
                
                # 4. Prepare Context for LLM
                # We inject the current state so the LLM knows what it has and what it needs
                state_context = f"""
[SYSTEM CONTEXT - DATA COLLECTED]
Current Stage: {stage_val}
Collected Data: {json.dumps({k: v for k, v in user_master_agent.state['entities'].items() if v}, indent=2)}
Missing Required Fields: {list(user_master_agent.state['missing_fields'])}
Missing KYC Fields: {list(user_master_agent.state['missing_kyc_fields'])}
Fraud Check Passed: {user_master_agent.state.get('fraud_check_passed', False)}
[INSTRUCTION]
If the user provides missing information, acknowledge it.
If all required fields are present (Missing Required Fields is empty), inform the user you are proceeding to check eligibility.
"""
                final_system_prompt = f"{system_prompt}\n{state_context}"

                # 5. Generate Response with History
                # Pass history excluding current message (which is passed as user_input)
                llm_resp = llm_service.generate_response(
                    user_input, 
                    final_system_prompt, 
                    chat_history=session['chat_history'][-10:] # Keep last 10 turns context
                )
                
                # Store user message in history
                session['chat_history'].append({"role": "user", "content": user_input})
                
                if llm_resp.get("intent") != "error":
                    # Store assistant response in history
                    if llm_resp.get("message"):
                        session['chat_history'].append({"role": "assistant", "content": llm_resp["message"]})
                    
                    # Map to existing schema
                    response = {
                        "message": llm_resp["message"],
                        "suggestions": llm_resp.get("suggestions", []),
                        "worker": worker_to_trigger, # Use backend-determined worker
                        "action": "none",
                        "intent": "llm_response",
                        "stage": stage_val,
                        "session_id": session_id,
                    }
                else:
                    # Fallback to backend on catastrophic failure
                    response = user_master_agent.handle(user_input)
            except Exception as e:
                print(f"Gemini Enabled Mode Error: {e}")
                response = user_master_agent.handle(user_input)

        # Mode 2: Hybrid (Orchestration)
        elif gemini_mode == "hybrid":
            try:
                # 1. Determine intent using backend
                intent, confidence = user_master_agent.detect_intent(user_input)
                
                # 2. Define "generative" vs "process" intents
                # Generative: greeting, help, unclear, maybe open-ended queries
                generative_intents = [
                    IntentType.GREETING, 
                    IntentType.HELP_GENERAL, 
                    IntentType.UNCLEAR,
                    IntentType.EXIT
                ]
                
                if intent in generative_intents:
                    # Use AI for natural language generation
                    context_prompt = system_prompt + f"\n[Context: Detected Intent '{intent.value}']"
                    
                    llm_resp = llm_service.generate_response(user_input, context_prompt)
                    
                    if llm_resp.get("intent") != "error":
                        response = {
                            "message": llm_resp["message"],
                            "suggestions": llm_resp.get("suggestions", []),
                            "worker": "none",
                            "action": "none",
                            "intent": intent.value,
                            "stage": user_master_agent.state["stage"].value,
                            "session_id": session_id
                        }
                    else:
                         response = user_master_agent.handle(user_input)
                else:
                    # Deterministic tasks (Loan App, KYC, Rates) -> Backend
                    response = user_master_agent.handle(user_input)

            except Exception as e:
                print(f"OpenRouter Hybrid Mode Error: {e}")
                response = user_master_agent.handle(user_input)

        # Mode 3: Disabled (Default to existing backend)
        else:
            response = user_master_agent.handle(user_input)
            
        # Ensure response is set (safety fallback)
        if response is None:
             response = user_master_agent.handle(user_input)
        
        # Update session state
        session['last_state'] = user_master_agent.state.copy()
        
        # Check if worker needs to be called
        worker_name = response.get('worker')
        
        # Convert worker names to actions
        action_map = {
            'underwriting': 'call_underwriting_api',
            'sales': 'call_sales_api',
            'fraud': 'call_fraud_api',
            'documentation': 'call_documentation_api'
        }
        
        if worker_name in action_map:
            response['action'] = action_map[worker_name]
            response['session_id'] = session_id
        
        # Add session info to response
        response['session_id'] = session_id
        response['interaction_count'] = session['interaction_count']

        # Log assistant response
        try:
            log_chat_event(session_id, 'message', {
                'role': 'assistant',
                'text': response.get('message'),
                'status': None,
                'details': {
                    'worker': response.get('worker'),
                    'action': response.get('action')
                }
            })
        except Exception:
            pass

        resp = jsonify(response)
        resp.headers['X-Session-ID'] = session_id
        return resp
        
    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({
            'message': 'Sorry, I encountered an error. Please try again.',
            'error': 'server_error',
            'worker': 'none'
        }), 500

@app.route('/underwrite', methods=['POST'])
def underwrite():
    """
    Worker endpoint for underwriting process.
    """
    try:
        session_id = request.headers.get('X-Session-ID')

        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'Invalid or expired session.'}), 400
        
        session = user_sessions[session_id]
        update_session_activity(session_id)
        
        user_master_agent = session['master_agent']
        current_state = user_master_agent.state
        
        # Step 1: Run fraud check first
        fraud_result = fraud_agent.perform_fraud_check(current_state['entities'])
        
        # Update master agent with fraud result
        user_master_agent.set_fraud_result(
            fraud_score=fraud_result.get('fraud_score', 0),
            fraud_flag=fraud_result.get('fraud_flag', 'Low')
        )
        
        # Step 2: If high fraud risk, reject immediately
        if fraud_result.get('fraud_flag') == 'High':
            user_master_agent.set_underwriting_result(
                risk_score=999,
                approval_status=False,
                interest_rate=0.0
            )
            
            session['fraud_result'] = fraud_result
            session['last_state'] = user_master_agent.state.copy()
            
            return jsonify({
                'message': 'Application rejected due to verification issues.',
                'approval_status': False,
                'reason': 'fraud_detected',
                'fraud_details': fraud_result,
                'worker': 'none',
                'next_action': 'terminate'
            })
        
        # Step 3: Proceed with underwriting (only if fraud is not High)
        underwriting_result = underwriting_agent.perform_underwriting(
            current_state['entities'],
            fraud_score=fraud_result.get('fraud_score', 0)
        )
        
        # Step 4: Update master agent with underwriting result
        user_master_agent.set_underwriting_result(
            risk_score=underwriting_result['risk_score'],
            approval_status=underwriting_result['approval_status'],
            interest_rate=underwriting_result.get('interest_rate', 12.5)  # Default
        )
        
        # Step 5: Generate response based on result
        if underwriting_result['approval_status']:
            response = {
                'message': 'Your application has been pre-approved!',
                'approval_status': True,
                'risk_score': underwriting_result['risk_score'],
                'interest_rate': underwriting_result.get('interest_rate', 12.5),
                'worker': 'sales',
                'action': 'call_sales_api'
            }
        else:
            response = {
                'message': 'Unfortunately, your application was not approved at this time.',
                'approval_status': False,
                'reason': underwriting_result.get('reason', 'risk_assessment'),
                'worker': 'sales',  # Route to sales for counseling
                'action': 'call_sales_api'
            }
        
        # Update session
        session['underwriting_result'] = underwriting_result
        session['fraud_result'] = fraud_result
        session['last_state'] = user_master_agent.state.copy()

        # Log status change
        try:
            log_chat_event(session_id, 'status_change', {
                'role': 'system',
                'text': None,
                'status': 'approved' if underwriting_result['approval_status'] else 'rejected',
                'details': {
                    'risk_score': underwriting_result.get('risk_score'),
                    'interest_rate': underwriting_result.get('interest_rate'),
                    'fraud_score': fraud_result.get('fraud_score')
                }
            })
        except Exception:
            pass
        
        response['session_id'] = session_id
        resp = jsonify(response)
        resp.headers['X-Session-ID'] = session_id
        return resp
        
    except Exception as e:
        print(f"Error in /underwrite: {e}")
        return jsonify({
            'error': 'underwriting_failed',
            'message': 'Underwriting process failed. Please try again.'
        }), 500

@app.route('/sales', methods=['POST'])
def sales_negotiate():
    """
    Worker endpoint for sales and negotiation.
    """
    try:
        session_id = request.headers.get('X-Session-ID')
        
        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'Invalid or expired session.'}), 400
        
        session = user_sessions[session_id]
        update_session_activity(session_id)
        
        user_master_agent = session['master_agent']
        current_state = user_master_agent.state
        
        # Check stage to determine what kind of sales interaction is needed
        if current_state.get('stage') == 'offer':
            # Generate loan offer
            sales_offer = sales_agent.generate_offer(
                master_agent_state=current_state,
                negotiation_request=request.get_json().get('negotiate', False)
            )
            
            # Update master agent with offer
            user_master_agent.set_offer(sales_offer)
            
            response = {
                **sales_offer,
                'session_id': session_id,
                'stage': 'offer_presented'
            }
            
        elif current_state.get('stage') == 'rejection_counseling':
            # Provide counseling for rejected application
            counseling_response = sales_agent.provide_counseling(current_state)
            
            response = {
                'message': counseling_response,
                'session_id': session_id,
                'stage': 'counseling',
                'next_steps': [
                    'Improve credit score',
                    'Reduce existing debt',
                    'Reapply in 6 months'
                ]
            }
        
        else:
            response = {
                'message': 'I need more information to provide an offer.',
                'session_id': session_id,
                'worker': 'none'
            }
        
        session['last_state'] = user_master_agent.state.copy()

        resp = jsonify(response)
        resp.headers['X-Session-ID'] = session_id
        return resp
        
    except Exception as e:
        print(f"Error in /sales: {e}")
        return jsonify({
            'error': 'sales_processing_failed',
            'message': 'Failed to process sales request.'
        }), 500

@app.route('/fraud', methods=['POST'])
def fraud_check():
    """
    Dedicated endpoint for fraud detection.
    """
    try:
        session_id = request.headers.get('X-Session-ID')
        
        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'Invalid or expired session.'}), 400
        
        session = user_sessions[session_id]
        update_session_activity(session_id)
        
        user_master_agent = session['master_agent']
        current_state = user_master_agent.state
        
        # Perform fraud check
        fraud_result = fraud_agent.perform_fraud_check(current_state['entities'])
        
        # Update master agent
        user_master_agent.set_fraud_result(
            fraud_score=fraud_result['fraud_score'],
            fraud_flag=fraud_result['fraud_flag']
        )
        
        # Store in session
        session['fraud_result'] = fraud_result
        session['last_state'] = user_master_agent.state.copy()
        
        response = {
            'fraud_check': fraud_result,
            'session_id': session_id,
            'passed': fraud_result['fraud_flag'] != 'High'
        }
        
        if fraud_result['fraud_flag'] == 'High':
            response['message'] = 'Fraud check failed. Application cannot proceed.'
            response['worker'] = 'none'
            response['next_action'] = 'terminate'
            try:
                log_chat_event(session_id, 'status_change', {
                    'role': 'system',
                    'text': None,
                    'status': 'rejected',
                    'details': fraud_result
                })
            except Exception:
                pass
        else:
            response['message'] = 'Fraud check passed.'
            response['worker'] = 'underwriting'
            response['action'] = 'call_underwriting_api'
            try:
                log_chat_event(session_id, 'action', {
                    'role': 'system',
                    'text': 'fraud_check_passed',
                    'status': None,
                    'details': fraud_result
                })
            except Exception:
                pass
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in /fraud: {e}")
        return jsonify({
            'error': 'fraud_check_failed',
            'message': 'Fraud detection failed.'
        }), 500

@app.route('/documentation', methods=['POST'])
def documentation():
    """
    Final step: Generate sanction letter.
    """
    try:
        session_id = request.headers.get('X-Session-ID')
        
        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'Invalid or expired session.'}), 400
        
        session = user_sessions[session_id]
        user_master_agent = session['master_agent']
        current_state = user_master_agent.state
        
        # Verify conditions
        if not current_state.get('offer_accepted', False):
            return jsonify({
                'error': 'offer_not_accepted',
                'message': 'Please accept the offer first.'
            }), 400
        
        if not all(current_state['entities'].get(field) for field in ['pan', 'aadhaar', 'address']):
            return jsonify({
                'error': 'kyc_incomplete',
                'message': 'KYC details are incomplete.'
            }), 400
        
        # Generate sanction letter (text + PDF)
        letter_data = generate_sanction_letter_text(current_state)
        pdf_path = generate_sanction_pdf(current_state)
        
        # Update final state
        user_master_agent.state['stage'] = 'closed'
        user_master_agent.state['sanction_letter'] = letter_data['metadata']['sanction_id']
        user_master_agent.state['letter_generated_at'] = datetime.now().isoformat()
        
        # Update session
        session['sanction_letter'] = letter_data
        session['sanction_letter_pdf'] = pdf_path
        session['last_state'] = user_master_agent.state.copy()
        session['completed_at'] = datetime.now().isoformat()

        resp = jsonify({
            'message': 'Sanction letter generated successfully!',
            'letter_content': letter_data['content'],
            'metadata': letter_data['metadata'],
            'session_id': session_id,
            'stage': 'completed',
            'download_url': f'/download/{session_id}',  # Mock download URL
            'next_action': 'download_letter'
        })

        resp.headers['X-Session-ID'] = session_id
        try:
            log_chat_event(session_id, 'action', {
                'role': 'system',
                'text': 'sanction_letter_generated',
                'status': 'approved',
                'details': letter_data['metadata']
            })
        except Exception:
            pass
        return resp
        
    except Exception as e:
        print(f"Error in /documentation: {e}")
        return jsonify({
            'error': 'documentation_failed',
            'message': 'Failed to generate sanction letter.'
        }), 500

@app.route('/download/<session_id>', methods=['GET'])
def download_sanction_letter(session_id):
    """Download the generated sanction letter PDF for a session."""
    session = user_sessions.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    pdf_path = session.get('sanction_letter_pdf')
    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({'error': 'No sanction letter available for download'}), 404

    directory, filename = os.path.split(pdf_path)
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session status (for debugging)."""
    if session_id in user_sessions:
        session = user_sessions[session_id]
        # Don't expose full agent state, just summary
        return jsonify({
            'session_id': session_id,
            'created_at': session.get('created_at'),
            'last_activity': session.get('last_activity'),
            'interaction_count': session.get('interaction_count', 0),
            'current_stage': session.get('master_agent').state.get('stage'),
            'has_offer': session.get('master_agent').state.get('offer_accepted', False)
        })
    return jsonify({'error': 'Session not found'}), 404

@app.route('/reset/<session_id>', methods=['POST'])
def reset_session(session_id):
    """Reset a session."""
    if session_id in user_sessions:
        user_sessions[session_id] = {
            'master_agent': MasterAgent(),
            'last_activity': time.time(),
            'created_at': datetime.now().isoformat(),
            'interaction_count': 0
        }
        return jsonify({'message': 'Session reset successfully.'})
    return jsonify({'error': 'Session not found'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(user_sessions),
        'agents': ['master', 'underwriting', 'sales', 'fraud']
    })


# --- File uploads serving (for admin) ---
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- Admin helpers ---
def is_admin_logged_in():
    return session.get('admin_username') is not None


@app.route('/bank/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        admins = read_csv(ADMINS_CSV)
        for adm in admins:
            if adm.get('username') == username and adm.get('password') == password:
                session['admin_username'] = username
                return redirect(url_for('admin_dashboard'))
        return render_template('admin_login.html', error='Invalid credentials')
    return render_template('admin_login.html')


@app.route('/bank/admin/logout')
def admin_logout():
    session.clear()
    return redirect(url_for('admin_login'))


@app.route('/bank/admin/dashboard')
def admin_dashboard():
    if not is_admin_logged_in():
        return redirect(url_for('admin_login'))
    applications = read_csv(APPLICATIONS_CSV)
    return render_template('admin_dashboard.html', applications=applications)


@app.route('/bank/admin/update_application', methods=['POST'])
def update_application():
    if not is_admin_logged_in():
        return redirect(url_for('admin_login'))

    session_id = request.form.get('session_id')
    timestamp = request.form.get('timestamp')
    new_status = request.form.get('status')
    reason = request.form.get('rejection_reason', '')

    rows = read_csv(APPLICATIONS_CSV)
    if rows:
        headers = list(rows[0].keys())
    else:
        headers = [
            'timestamp', 'session_id', 'full_name', 'phone', 'email',
            'city', 'loan_amount', 'status', 'rejection_reason', 'attached_files'
        ]

    for r in rows:
        if r.get('session_id') == session_id and r.get('timestamp') == timestamp:
            r['status'] = new_status
            r['rejection_reason'] = reason
            break

    ensure_csv(APPLICATIONS_CSV, headers)
    import csv as _csv
    with open(APPLICATIONS_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = _csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    log_chat_event(session_id, 'status_change', {
        'role': 'admin',
        'text': None,
        'status': new_status,
        'details': {'rejection_reason': reason},
    })

    return redirect(url_for('admin_dashboard'))


@app.route('/bank/admin/tune', methods=['GET', 'POST'])
def admin_tune():
    if not is_admin_logged_in():
        return redirect(url_for('admin_login'))

    headers = ['timestamp', 'admin', 'type', 'content', 'filename']
    if request.method == 'POST':
        description = request.form.get('description', '').strip()
        upload = request.files.get('file')
        filename = ''
        if upload and upload.filename:
            filename = secure_filename(upload.filename)
            upload.save(os.path.join(UPLOADS_DIR, filename))

        row = {
            'timestamp': datetime.utcnow().isoformat(),
            'admin': session.get('admin_username'),
            'type': 'text_and_file',
            'content': description,
            'filename': filename,
        }
        append_csv(TUNING_CSV, row, headers)

        return redirect(url_for('admin_tune'))

    items = read_csv(TUNING_CSV)
    return render_template('admin_tune.html', items=items)


@app.route('/bank/admin/tune/clear', methods=['POST'])
def admin_tune_clear():
    if not is_admin_logged_in():
        return redirect(url_for('admin_login'))

    headers = ['timestamp', 'admin', 'type', 'content', 'filename']
    # Clear file by writing just headers
    with open(TUNING_CSV, mode='w', newline='', encoding='utf-8') as f:
        import csv as _csv
        writer = _csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

    return redirect(url_for('admin_tune'))


@app.route('/bank/admin/logs')
def admin_logs():
    if not is_admin_logged_in():
        return redirect(url_for('admin_login'))
    logs = read_csv(CHAT_LOGS_CSV)
    return render_template('admin_logs.html', logs=logs)


# --- Application submission (for dashboard) ---
@app.route('/api/widget/submit_application', methods=['POST'])
def submit_application():
    session_id = request.headers.get('X-Session-ID') or f"session_{uuid.uuid4().hex[:16]}"
    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({'error': 'Invalid request'}), 400

    headers = [
        'timestamp', 'session_id', 'full_name', 'phone', 'email',
        'city', 'loan_amount', 'status', 'rejection_reason', 'attached_files',
    ]
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'session_id': session_id,
        'full_name': data.get('full_name', ''),
        'phone': data.get('phone', ''),
        'email': data.get('email', ''),
        'city': data.get('city', ''),
        'loan_amount': data.get('loan_amount', ''),
        'status': 'under_review',
        'rejection_reason': '',
        'attached_files': json.dumps(data.get('files', []), ensure_ascii=False),
    }
    append_csv(APPLICATIONS_CSV, row, headers)

    log_chat_event(session_id, 'action', {
        'role': 'user',
        'text': 'submit_application',
        'status': 'under_review',
        'details': row,
    })

    return jsonify({'ok': True, 'session_id': session_id})


# --- File upload endpoint for widget/applications ---
@app.route('/api/widget/upload', methods=['POST'])
def widget_upload():
    session_id = request.headers.get('X-Session-ID') or f"session_{uuid.uuid4().hex[:16]}"
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{session_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}")
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        log_chat_event(session_id, 'action', {
            'role': 'user',
            'text': 'upload_file',
            'status': None,
            'details': {'filename': filename},
        })
        return jsonify({'filename': filename, 'url': url_for('uploaded_file', filename=filename, _external=False), 'session_id': session_id})

    return jsonify({'error': 'Invalid file type'}), 400


@app.template_filter('load_json')
def load_json_filter(s):
    try:
        return json.loads(s)
    except Exception:
        return []

if __name__ == '__main__':
    print("=" * 60)
    print("CREDGEN Loan Application System")
    print("=" * 60)
    print("Server starting on http://0.0.0.0:5000")
    print("Available endpoints:")
    print("  GET  /               - Frontend page (index.html)")
    print("  GET  /frontend/*     - Frontend static files (CSS, JS)")
    print("  POST /chat           - Main conversation endpoint")
    print("  POST /underwrite     - Underwriting process")
    print("  POST /sales          - Sales and negotiation")
    print("  POST /fraud          - Fraud detection")
    print("  POST /documentation  - Generate sanction letter")
    print("  GET  /health         - Health check")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
