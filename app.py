from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email
import json
import os
import re
import time
import uuid
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize OpenAI-compatible client for Synthetic API
client = OpenAI(
    api_key=os.environ.get("SYNTHETIC_API_KEY"),
    base_url="https://api.synthetic.new/openai/v1"
)

# Simple rate limiting (in-memory)
rate_limit_cache = {}

# Metrics store (in-memory for demo)
metrics = {
    'total_requests': 0,
    'successful_emails': 0,
    'failed_emails': 0,
    'llm_calls': 0,
    'llm_errors': 0,
    'llm_fallbacks': 0,
    'avg_latency_ms': 0,
    'total_latency_ms': 0,
    'traces': []
}

def scrub_secrets(text):
    """Remove sensitive data from text for logging."""
    if not text:
        return text
    # Scrub email addresses
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', str(text))
    # Scrub API keys
    text = re.sub(r'(api[_-]?key|token|secret)["\s:=]+["\']?[\w-]{20,}["\']?', r'\1=[REDACTED]', str(text), flags=re.IGNORECASE)
    return text

def create_trace(trace_id, buyer_name):
    """Create a new trace for observability."""
    return {
        'trace_id': trace_id,
        'buyer_name': buyer_name,
        'start_time': datetime.utcnow().isoformat(),
        'steps': [],
        'llm_calls': [],
        'total_duration_ms': 0,
        'status': 'in_progress'
    }

def add_trace_step(trace, step_name, duration_ms, status='success', details=None):
    """Add a step to the trace."""
    trace['steps'].append({
        'step': step_name,
        'duration_ms': round(duration_ms, 2),
        'status': status,
        'timestamp': datetime.utcnow().isoformat(),
        'details': details
    })

def log_llm_call(trace, call_type, input_text, output_text, duration_ms, tokens_used=None):
    """Log LLM input/output with secrets scrubbed."""
    trace['llm_calls'].append({
        'type': call_type,
        'input_preview': scrub_secrets(input_text[:500] + '...' if len(input_text) > 500 else input_text),
        'output_preview': scrub_secrets(output_text[:500] + '...' if len(output_text) > 500 else output_text),
        'input_length': len(input_text),
        'output_length': len(output_text),
        'duration_ms': round(duration_ms, 2),
        'tokens_used': tokens_used,
        'timestamp': datetime.utcnow().isoformat()
    })

def load_properties():
    """Load property listings from JSON file"""
    properties_path = os.path.join(os.path.dirname(__file__), 'data', 'properties.json')
    with open(properties_path, 'r') as f:
        return json.load(f)

def load_prompt(prompt_name):
    """Load a prompt template from the prompts directory"""
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', f'{prompt_name}.txt')
    with open(prompt_path, 'r') as f:
        return f.read()

def analyze_buyer_preferences(buyer_input: str) -> dict:
    """Use LLM to extract structured preferences from buyer input."""
    prompt = load_prompt('analyzer')

    try:
        response = client.chat.completions.create(
            model="hf:zai-org/GLM-4.5",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(buyer_input=buyer_input)
                }
            ]
        )

        if not response.choices or not response.choices[0].message:
            print("No response from model")
            return parse_preferences_manually(buyer_input)

        response_text = response.choices[0].message.content
        if not response_text:
            print("Empty response content")
            return parse_preferences_manually(buyer_input)

        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            parsed = json.loads(response_text[start:end])
            if parsed:
                return parsed

        print(f"No JSON found in response: {response_text[:200]}")
        return parse_preferences_manually(buyer_input)

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return parse_preferences_manually(buyer_input)
    except Exception as e:
        print(f"Error in analyze_buyer_preferences: {e}")
        return parse_preferences_manually(buyer_input)

def parse_preferences_manually(buyer_input: str) -> dict:
    """Fallback parser to extract basic preferences from text."""
    preferences = {}
    text = buyer_input.lower()

    # Extract budget
    import re
    budget_match = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|mil)', text)
    if budget_match:
        amount = float(budget_match.group(1).replace(',', ''))
        preferences['max_budget'] = int(amount * 1000000)
    else:
        budget_match = re.search(r'\$?([\d,]+)(?:k)?', text)
        if budget_match:
            amount = int(budget_match.group(1).replace(',', ''))
            if amount < 10000:
                preferences['max_budget'] = amount * 1000
            else:
                preferences['max_budget'] = amount

    # Extract bedrooms
    bed_match = re.search(r'(\d+)\s*(?:bed|bedroom|br)', text)
    if bed_match:
        preferences['min_bedrooms'] = int(bed_match.group(1))

    # Extract features
    features = []
    feature_keywords = ['backyard', 'garage', 'pool', 'view', 'modern kitchen', 'fireplace', 'home office']
    for feature in feature_keywords:
        if feature in text:
            features.append(feature)
    if features:
        preferences['desired_features'] = features

    # Extract locations (common SF and Bay Area neighborhoods)
    locations = []
    neighborhoods = ['pacific heights', 'marina', 'noe valley', 'mission', 'soma', 'hayes valley',
                    'russian hill', 'nopa', 'richmond', 'sunset', 'fremont', 'oakland', 'berkeley',
                    'san jose', 'santa clara', 'sunnyvale', 'mountain view', 'palo alto']
    for hood in neighborhoods:
        if hood in text:
            locations.append(hood.title())
    if locations:
        preferences['preferred_locations'] = locations

    # Extract property type preferences
    if any(term in text for term in ['multi-unit', 'multi unit', 'multiplex', 'duplex', 'triplex', 'fourplex', 'apartment building', 'units']):
        preferences['property_type'] = 'multi-unit'
        if 'desired_features' not in preferences:
            preferences['desired_features'] = []
        preferences['desired_features'].append('multi-unit')

    # Extract investment-related preferences
    if any(term in text for term in ['invest', 'roi', 'cap rate', 'rental', 'income', 'cash flow']):
        if 'desired_features' not in preferences:
            preferences['desired_features'] = []
        preferences['desired_features'].extend(['investment property', 'high ROI', 'good ROI'])

    return preferences

def match_properties(preferences: dict, properties: list) -> list:
    """Match properties against buyer preferences."""
    matches = []

    for prop in properties:
        score = 15  # Base score for all properties
        reasons = []

        max_budget = preferences.get('max_budget')
        if max_budget and 'price' in prop:
            if prop['price'] <= max_budget:
                score += 40
                reasons.append("Within budget")
            elif prop['price'] <= max_budget * 1.1:
                score += 10
                reasons.append("Slightly over budget")
        elif not max_budget:
            score += 20  # No budget specified

        min_bedrooms = preferences.get('min_bedrooms')
        if min_bedrooms and 'bedrooms' in prop:
            if prop['bedrooms'] >= min_bedrooms:
                score += 15
                reasons.append(f"{prop['bedrooms']} bedrooms meets requirement")

        if 'preferred_locations' in preferences and 'neighborhood' in prop:
            if prop['neighborhood'].lower() in [loc.lower() for loc in preferences.get('preferred_locations', [])]:
                score += 20
                reasons.append(f"In preferred area: {prop['neighborhood']}")

        if 'desired_features' in preferences and 'features' in prop:
            matching_features = set(f.lower() for f in preferences.get('desired_features', [])) & \
                              set(f.lower() for f in prop.get('features', []))
            if matching_features:
                # High value for matching features
                score += len(matching_features) * 25
                reasons.append(f"Has: {', '.join(matching_features)}")

        # Always add property with at least a base reason
        if not reasons:
            reasons.append("Available listing")

        matches.append({
            'property': prop,
            'score': score,
            'match_reasons': reasons
        })

    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches[:5]

def generate_email(buyer_name: str, preferences: dict, matches: list) -> str:
    """Use LLM to generate a personalized email with property recommendations."""
    prompt = load_prompt('email_gen')

    matches_text = ""
    for i, match in enumerate(matches, 1):
        prop = match['property']
        matches_text += f"\n{i}. {prop.get('address', 'Address TBD')}\n"
        matches_text += f"   Price: ${prop.get('price', 0):,}\n"
        matches_text += f"   Bedrooms: {prop.get('bedrooms', 'N/A')} | Bathrooms: {prop.get('bathrooms', 'N/A')}\n"
        matches_text += f"   Neighborhood: {prop.get('neighborhood', 'N/A')}\n"
        matches_text += f"   Features: {', '.join(prop.get('features', []))}\n"
        matches_text += f"   Why it matches: {', '.join(match['match_reasons'])}\n"

    response = client.chat.completions.create(
        model="hf:zai-org/GLM-4.5",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": prompt.format(
                    buyer_name=buyer_name,
                    preferences=json.dumps(preferences, indent=2),
                    matches=matches_text
                )
            }
        ]
    )

    return response.choices[0].message.content or "Email generation failed"

def run_campaign(buyer_name: str, buyer_input: str) -> dict:
    """Main agent workflow."""
    try:
        properties = load_properties()
        preferences = analyze_buyer_preferences(buyer_input)
        matches = match_properties(preferences, properties)

        # Preferences may be empty if LLM failed to parse, but we still have matches

        return {
            'status': 'success',
            'buyer_name': buyer_name,
            'preferences': preferences,
            'matches': [
                {
                    'address': m['property'].get('address'),
                    'price': m['property'].get('price'),
                    'score': m['score'],
                    'reasons': m['match_reasons'],
                    'image_url': m['property'].get('image_url'),
                    'bedrooms': m['property'].get('bedrooms'),
                    'bathrooms': m['property'].get('bathrooms'),
                    'sqft': m['property'].get('sqft'),
                    'description': m['property'].get('description'),
                    'features': m['property'].get('features', [])
                }
                for m in matches
            ],
            'generated_at': datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {
            'status': 'error',
            'buyer_name': buyer_name,
            'error': str(e)
        }

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate email campaign from buyer input"""
    try:
        data = request.json
        buyer_name = data.get('buyer_name', 'Valued Client')
        buyer_input = data.get('buyer_input', '')

        if not buyer_input:
            return jsonify({'error': 'Buyer input is required'}), 400

        ip = request.remote_addr
        if ip in rate_limit_cache:
            rate_limit_cache[ip] += 1
            if rate_limit_cache[ip] > 20:
                return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
        else:
            rate_limit_cache[ip] = 1

        result = run_campaign(buyer_name, buyer_input)
        return jsonify(result)

    except Exception as e:
        print(f"Error generating campaign: {e}")
        return jsonify({'error': 'An error occurred generating the campaign'}), 500

@app.route('/generate-email', methods=['POST'])
def generate_email_endpoint():
    """Generate email for selected properties"""
    try:
        data = request.json
        buyer_name = data.get('buyer_name', 'Valued Client')
        preferences = data.get('preferences', {})
        selected_matches = data.get('selected_matches', [])

        if len(selected_matches) != 2:
            return jsonify({'error': 'Please select exactly 2 properties'}), 400

        # Format matches for email generation
        matches_text = ""
        for i, match in enumerate(selected_matches, 1):
            matches_text += f"\n{i}. {match.get('address', 'Address TBD')}\n"
            matches_text += f"   Price: ${match.get('price', 0):,}\n"
            matches_text += f"   Bedrooms: {match.get('bedrooms', 'N/A')} | Bathrooms: {match.get('bathrooms', 'N/A')}\n"
            matches_text += f"   Sqft: {match.get('sqft', 'N/A')}\n"
            matches_text += f"   Features: {', '.join(match.get('features', []))}\n"
            matches_text += f"   Description: {match.get('description', '')}\n"
            matches_text += f"   Image: {match.get('image_url', '')}\n"
            matches_text += f"   Why it matches: {', '.join(match.get('reasons', []))}\n"

        prompt = load_prompt('email_gen')

        response = client.chat.completions.create(
            model="hf:zai-org/GLM-4.5",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(
                        buyer_name=buyer_name,
                        preferences=json.dumps(preferences, indent=2),
                        matches=matches_text
                    )
                }
            ]
        )

        email_content = response.choices[0].message.content or "Email generation failed"

        return jsonify({
            'status': 'success',
            'email': email_content
        })

    except Exception as e:
        print(f"Error generating email: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'scout-ai'})

@app.route('/metrics')
def get_metrics():
    """Get observability metrics"""
    return jsonify({
        'metrics': {
            'total_requests': metrics['total_requests'],
            'successful_emails': metrics['successful_emails'],
            'failed_emails': metrics['failed_emails'],
            'success_rate': round(metrics['successful_emails'] / max(metrics['total_requests'], 1) * 100, 2),
            'llm_calls': metrics['llm_calls'],
            'llm_errors': metrics['llm_errors'],
            'llm_fallbacks': metrics['llm_fallbacks'],
            'avg_latency_ms': round(metrics['total_latency_ms'] / max(metrics['total_requests'], 1), 2)
        },
        'recent_traces': metrics['traces'][-10:],  # Last 10 traces
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/run-agent-cycle', methods=['POST'])
def run_agent_cycle():
    """Run full agent cycle for one buyer: analyze -> match -> generate -> send email"""
    cycle_start = time.time()
    trace_id = str(uuid.uuid4())[:8]
    trace = create_trace(trace_id, 'buyer')

    metrics['total_requests'] += 1
    logger.info(f"[{trace_id}] Starting agent cycle")

    try:
        data = request.json
        buyer_name = data.get('buyer_name', 'Valued Client')
        buyer_email = data.get('buyer_email')
        buyer_input = data.get('buyer_input', '')
        personalization_style = data.get('personalization_style', '')
        exclude_addresses = data.get('exclude_addresses', [])  # For sending different properties in subsequent emails

        trace['buyer_name'] = buyer_name

        if not buyer_input:
            metrics['failed_emails'] += 1
            return jsonify({'error': 'Buyer input is required', 'step': 'validation', 'trace_id': trace_id}), 400

        if not buyer_email:
            metrics['failed_emails'] += 1
            return jsonify({'error': 'Buyer email is required', 'step': 'validation', 'trace_id': trace_id}), 400

        steps = []

        # Step 1: Load properties
        step_start = time.time()
        properties = load_properties()
        step_duration = (time.time() - step_start) * 1000
        add_trace_step(trace, 'load_properties', step_duration, details={'count': len(properties)})
        steps.append({'action': 'Loading property database', 'status': 'complete', 'duration_ms': round(step_duration, 2)})
        logger.info(f"[{trace_id}] Loaded {len(properties)} properties in {step_duration:.2f}ms")

        # Step 2: Analyze preferences (LLM call)
        step_start = time.time()
        llm_input = f"Buyer input: {buyer_input}"
        preferences = analyze_buyer_preferences(buyer_input)
        step_duration = (time.time() - step_start) * 1000
        metrics['llm_calls'] += 1

        # Check if fallback was used
        if not preferences or len(preferences) == 0:
            metrics['llm_fallbacks'] += 1
            logger.warning(f"[{trace_id}] LLM fallback triggered for preference analysis")

        log_llm_call(trace, 'preference_analysis', llm_input, json.dumps(preferences), step_duration)
        add_trace_step(trace, 'analyze_preferences', step_duration, details={'preferences_extracted': len(preferences)})
        steps.append({'action': f'Analyzing preferences for {buyer_name}', 'status': 'complete', 'duration_ms': round(step_duration, 2)})
        logger.info(f"[{trace_id}] Analyzed preferences in {step_duration:.2f}ms")

        # Step 3: Match properties
        step_start = time.time()
        matches = match_properties(preferences, properties)

        # Filter out excluded properties
        if exclude_addresses:
            matches = [m for m in matches if m['property'].get('address') not in exclude_addresses]

        step_duration = (time.time() - step_start) * 1000
        add_trace_step(trace, 'match_properties', step_duration, details={'matches_found': len(matches)})
        steps.append({'action': f'Found {len(matches)} matching properties', 'status': 'complete', 'duration_ms': round(step_duration, 2)})
        logger.info(f"[{trace_id}] Matched {len(matches)} properties in {step_duration:.2f}ms")

        if len(matches) < 2:
            return jsonify({
                'error': 'Not enough matching properties found',
                'step': 'matching',
                'steps': steps
            }), 400

        # Step 4: Auto-select top 2 properties
        selected_matches = [
            {
                'address': m['property'].get('address'),
                'price': m['property'].get('price'),
                'score': m['score'],
                'reasons': m['match_reasons'],
                'image_url': m['property'].get('image_url'),
                'bedrooms': m['property'].get('bedrooms'),
                'bathrooms': m['property'].get('bathrooms'),
                'sqft': m['property'].get('sqft'),
                'description': m['property'].get('description'),
                'features': m['property'].get('features', [])
            }
            for m in matches[:2]
        ]
        steps.append({'action': f'Selected top 2 properties (scores: {matches[0]["score"]}, {matches[1]["score"]})', 'status': 'complete'})

        # Step 5: Generate email (LLM call)
        step_start = time.time()
        matches_text = ""
        for i, match in enumerate(selected_matches, 1):
            matches_text += f"\n{i}. {match.get('address', 'Address TBD')}\n"
            matches_text += f"   Price: ${match.get('price', 0):,}\n"
            matches_text += f"   Bedrooms: {match.get('bedrooms', 'N/A')} | Bathrooms: {match.get('bathrooms', 'N/A')}\n"
            matches_text += f"   Sqft: {match.get('sqft', 'N/A')}\n"
            matches_text += f"   Features: {', '.join(match.get('features', []))}\n"
            matches_text += f"   Description: {match.get('description', '')}\n"
            matches_text += f"   Image: {match.get('image_url', '')}\n"
            matches_text += f"   Why it matches: {', '.join(match.get('reasons', []))}\n"

        prompt = load_prompt('email_gen')

        # Add personalization style to the prompt if provided
        prompt_content = prompt.format(
            buyer_name=buyer_name,
            preferences=json.dumps(preferences, indent=2),
            matches=matches_text
        )

        if personalization_style:
            prompt_content += f"\n\nPersonalization Style Instructions:\n{personalization_style}\n\nApply these style instructions when writing the email."

        response = client.chat.completions.create(
            model="hf:zai-org/GLM-4.5",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": prompt_content
                }
            ]
        )
        email_content = response.choices[0].message.content or "Email generation failed"
        step_duration = (time.time() - step_start) * 1000
        metrics['llm_calls'] += 1

        log_llm_call(trace, 'email_generation', prompt_content, email_content, step_duration)
        add_trace_step(trace, 'generate_email', step_duration, details={'output_length': len(email_content)})
        steps.append({'action': 'Generating personalized email', 'status': 'complete', 'duration_ms': round(step_duration, 2)})
        logger.info(f"[{trace_id}] Generated email ({len(email_content)} chars) in {step_duration:.2f}ms")

        # Step 6: Send email
        step_start = time.time()

        # Extract subject
        subject = "New Property Matches for You!"
        lines = email_content.split('\n')
        for line in lines:
            if line.lower().startswith('subject:'):
                subject = line.replace('Subject:', '').replace('subject:', '').strip()
                break

        # Check if content is HTML
        is_html = '<' in email_content and '>' in email_content

        # Create and send email
        from_email = Email(
            os.environ.get('SENDGRID_FROM_EMAIL', 'noreply@homematch.ai'),
            'Hack The Bay Realty Co. - Elly Lin'
        )
        if is_html:
            message = Mail(
                from_email=from_email,
                to_emails=buyer_email,
                subject=subject,
                html_content=email_content
            )
        else:
            message = Mail(
                from_email=from_email,
                to_emails=buyer_email,
                subject=subject,
                plain_text_content=email_content
            )

        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        sg.send(message)

        step_duration = (time.time() - step_start) * 1000
        add_trace_step(trace, 'send_email', step_duration, details={'recipient': scrub_secrets(buyer_email)})
        steps.append({'action': f'Sending email to {buyer_email}', 'status': 'complete', 'duration_ms': round(step_duration, 2)})
        steps.append({'action': f'Email delivered to {buyer_email}', 'status': 'complete'})
        logger.info(f"[{trace_id}] Sent email in {step_duration:.2f}ms")

        # Finalize trace
        total_duration = (time.time() - cycle_start) * 1000
        trace['total_duration_ms'] = round(total_duration, 2)
        trace['status'] = 'success'
        trace['end_time'] = datetime.utcnow().isoformat()

        # Update metrics
        metrics['successful_emails'] += 1
        metrics['total_latency_ms'] += total_duration
        metrics['traces'].append(trace)
        if len(metrics['traces']) > 100:  # Keep only last 100 traces
            metrics['traces'] = metrics['traces'][-100:]

        logger.info(f"[{trace_id}] Agent cycle completed in {total_duration:.2f}ms")

        return jsonify({
            'status': 'success',
            'buyer_name': buyer_name,
            'buyer_email': buyer_email,
            'properties_matched': len(matches),
            'properties_selected': [m['address'] for m in selected_matches],
            'steps': steps,
            'email_content': email_content,
            'timestamp': datetime.utcnow().isoformat(),
            'trace_id': trace_id,
            'total_duration_ms': round(total_duration, 2)
        })

    except Exception as e:
        # Update metrics for failure
        metrics['failed_emails'] += 1
        total_duration = (time.time() - cycle_start) * 1000
        trace['total_duration_ms'] = round(total_duration, 2)
        trace['status'] = 'error'
        trace['error'] = str(e)
        trace['end_time'] = datetime.utcnow().isoformat()
        metrics['traces'].append(trace)

        logger.error(f"[{trace_id}] Agent cycle failed: {e}")
        return jsonify({
            'error': str(e),
            'step': 'unknown',
            'steps': steps if 'steps' in locals() else [],
            'trace_id': trace_id
        }), 500

@app.route('/send-email', methods=['POST'])
def send_email():
    """Send email via SendGrid"""
    try:
        data = request.json
        to_email = data.get('to_email')
        to_name = data.get('to_name')
        email_content = data.get('email_content')

        if not to_email or not email_content:
            return jsonify({'error': 'Email and content are required'}), 400

        # Extract subject from email content (first line after "Subject:")
        subject = "New Property Matches for You!"
        lines = email_content.split('\n')
        for line in lines:
            if line.lower().startswith('subject:'):
                subject = line.replace('Subject:', '').replace('subject:', '').strip()
                break

        # Check if content is HTML
        is_html = '<' in email_content and '>' in email_content

        # Create SendGrid message
        from_email = Email(
            os.environ.get('SENDGRID_FROM_EMAIL', 'noreply@homematch.ai'),
            'Hack The Bay Realty Co. - Elly Lin'
        )
        if is_html:
            message = Mail(
                from_email=from_email,
                to_emails=to_email,
                subject=subject,
                html_content=email_content
            )
        else:
            message = Mail(
                from_email=from_email,
                to_emails=to_email,
                subject=subject,
                plain_text_content=email_content
            )

        # Send email
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)

        return jsonify({
            'status': 'success',
            'message': f'Email sent to {to_email}'
        })

    except Exception as e:
        print(f"Error sending email: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
