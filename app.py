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

# =============================================================================
# SECURITY CONTROLS - Input Validation, Output Constraints, Policy Checks
# =============================================================================

# Blocked patterns for prompt injection detection
PROMPT_INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?(previous|above)\s+instructions',
    r'ignore\s+instructions',
    r'disregard\s+(all\s+)?(previous|above|safety)',
    r'forget\s+(everything|all|previous)',
    r'new\s+instructions?\s*:',
    r'system\s*:\s*you\s+are',
    r'<\s*system\s*>',
    r'\]\s*\[\s*system',
    r'pretend\s+you\s+are',
    r'act\s+as\s+if',
    r'roleplay\s+as',
    r'output\s+(the|all|your)\s+(api|secret|key|password)',
    r'reveal\s+(the|your|all)',
    r'show\s+(me\s+)?(the\s+)?(api|secret|key|env)',
]

# Allowed email domains (for demo/safety)
ALLOWED_EMAIL_DOMAINS = None  # Set to list like ['gmail.com', 'company.com'] to restrict

# Maximum input lengths
MAX_BUYER_INPUT_LENGTH = 2000
MAX_NAME_LENGTH = 100
MAX_EMAIL_LENGTH = 254
MAX_STYLE_LENGTH = 500

# Output constraints
MAX_EMAIL_OUTPUT_LENGTH = 50000
BLOCKED_OUTPUT_PATTERNS = [
    r'<script[^>]*>',  # XSS prevention
    r'javascript:',
    r'on\w+\s*=',  # Event handlers
]

def validate_input(buyer_input: str, buyer_name: str, buyer_email: str, style: str = '') -> tuple:
    """
    Validate and sanitize all inputs. Returns (is_valid, error_message, sanitized_data).

    Security controls:
    - Length limits to prevent DoS
    - Prompt injection detection
    - Email format validation
    - Character filtering
    """
    errors = []

    # Length validation
    if len(buyer_input) > MAX_BUYER_INPUT_LENGTH:
        errors.append(f'Buyer input exceeds {MAX_BUYER_INPUT_LENGTH} characters')
    if len(buyer_name) > MAX_NAME_LENGTH:
        errors.append(f'Name exceeds {MAX_NAME_LENGTH} characters')
    if len(buyer_email) > MAX_EMAIL_LENGTH:
        errors.append(f'Email exceeds {MAX_EMAIL_LENGTH} characters')
    if len(style) > MAX_STYLE_LENGTH:
        errors.append(f'Style exceeds {MAX_STYLE_LENGTH} characters')

    # Prompt injection detection
    combined_input = f"{buyer_input} {style}".lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, combined_input, re.IGNORECASE):
            logger.warning(f"Prompt injection attempt detected: {pattern}")
            metrics['security_blocks'] = metrics.get('security_blocks', 0) + 1
            errors.append('Input contains disallowed patterns')
            break

    # Email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, buyer_email):
        errors.append('Invalid email format')

    # Domain allowlist check
    if ALLOWED_EMAIL_DOMAINS:
        domain = buyer_email.split('@')[-1].lower()
        if domain not in ALLOWED_EMAIL_DOMAINS:
            errors.append(f'Email domain not allowed')

    # Sanitize inputs (remove control characters)
    sanitized = {
        'buyer_input': re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', buyer_input),
        'buyer_name': re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', buyer_name),
        'buyer_email': buyer_email.strip().lower(),
        'style': re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', style)
    }

    if errors:
        return False, '; '.join(errors), None

    return True, None, sanitized

def validate_output(content: str) -> tuple:
    """
    Validate LLM output before sending. Returns (is_valid, sanitized_content).

    Security controls:
    - Length limits
    - XSS pattern blocking
    - Content sanitization
    """
    if len(content) > MAX_EMAIL_OUTPUT_LENGTH:
        logger.warning(f"Output exceeds max length: {len(content)}")
        content = content[:MAX_EMAIL_OUTPUT_LENGTH] + "\n\n[Content truncated for safety]"

    # Check for blocked patterns
    for pattern in BLOCKED_OUTPUT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            logger.warning(f"Blocked pattern in output: {pattern}")
            # Remove the pattern rather than blocking entirely
            content = re.sub(pattern, '[REMOVED]', content, flags=re.IGNORECASE)

    return True, content

def check_sendgrid_policy(recipient_email: str, content_length: int) -> tuple:
    """
    Policy check before SendGrid API call.
    Returns (allowed, reason).

    Implements least-privilege by validating before external API calls.
    """
    # Rate check per recipient (simple in-memory)
    recipient_key = f"sendgrid:{recipient_email}"
    current_count = rate_limit_cache.get(recipient_key, 0)

    if current_count >= 10:  # Max 10 emails per recipient per session
        logger.warning(f"SendGrid policy block: rate limit for {scrub_secrets(recipient_email)}")
        return False, 'Rate limit exceeded for this recipient'

    # Content size check
    if content_length > MAX_EMAIL_OUTPUT_LENGTH:
        return False, 'Email content too large'

    # Update rate counter
    rate_limit_cache[recipient_key] = current_count + 1

    return True, None

def check_llm_policy(input_length: int, call_type: str) -> tuple:
    """
    Policy check before LLM API call.
    Returns (allowed, reason).
    """
    # Check total LLM calls (prevent runaway costs)
    if metrics.get('llm_calls', 0) > 1000:
        logger.warning("LLM policy block: daily call limit reached")
        return False, 'LLM call limit reached'

    # Input length check
    max_input = 10000 if call_type == 'email_generation' else 5000
    if input_length > max_input:
        return False, f'Input too long for {call_type}'

    return True, None

def safe_fail(error_msg: str, trace_id: str = None, include_trace: bool = True) -> dict:
    """
    Generate safe failure response without leaking internal details.
    """
    response = {
        'status': 'error',
        'error': error_msg,
        'timestamp': datetime.utcnow().isoformat()
    }
    if trace_id and include_trace:
        response['trace_id'] = trace_id
    return response

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
    # Match millions (e.g., $1M, $1.5 million, 2mil)
    budget_match = re.search(r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|m|mil)\b', text, re.IGNORECASE)
    if budget_match:
        try:
            amount_str = budget_match.group(1).replace(',', '').strip()
            if amount_str:
                amount = float(amount_str)
                preferences['max_budget'] = int(amount * 1000000)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse million budget: {e}")

    # If no millions match, try thousands (e.g., $500k, $800K)
    if 'max_budget' not in preferences:
        budget_match = re.search(r'\$?([\d,]+)\s*k\b', text, re.IGNORECASE)
        if budget_match:
            try:
                amount_str = budget_match.group(1).replace(',', '').strip()
                if amount_str:
                    amount = int(amount_str)
                    preferences['max_budget'] = amount * 1000
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse k budget: {e}")

    # Try plain dollar amounts (e.g., $1,000,000)
    if 'max_budget' not in preferences:
        budget_match = re.search(r'\$([\d,]+)', text)
        if budget_match:
            try:
                amount_str = budget_match.group(1).replace(',', '').strip()
                if amount_str:
                    amount = int(amount_str)
                    preferences['max_budget'] = amount
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse dollar budget: {e}")

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

    # Extract locations (SF/Bay Area neighborhoods + major US cities)
    locations = []
    neighborhoods = ['pacific heights', 'marina', 'noe valley', 'mission', 'soma', 'hayes valley',
                    'russian hill', 'nopa', 'richmond', 'sunset', 'san francisco',
                    # Bay Area cities
                    'fremont', 'oakland', 'berkeley', 'san jose', 'santa clara', 'sunnyvale',
                    'mountain view', 'palo alto', 'san mateo', 'redwood city', 'daly city',
                    'south san francisco', 'hayward', 'concord', 'walnut creek', 'pleasanton',
                    'livermore', 'cupertino', 'milpitas', 'newark', 'union city', 'alameda',
                    'san leandro', 'san ramon', 'dublin', 'menlo park', 'foster city',
                    'burlingame', 'san bruno', 'pacifica', 'saratoga', 'los gatos', 'campbell',
                    'morgan hill', 'gilroy', 'vallejo', 'fairfield', 'napa', 'sonoma',
                    'petaluma', 'santa rosa', 'san rafael', 'novato', 'sausalito', 'tiburon',
                    # Major US cities
                    'boston', 'new york', 'chicago', 'los angeles', 'seattle', 'austin',
                    'denver', 'miami', 'atlanta', 'portland', 'phoenix', 'san diego',
                    'dallas', 'houston', 'philadelphia', 'washington dc', 'detroit',
                    'minneapolis', 'nashville', 'charlotte', 'raleigh', 'tampa',
                    'orlando', 'las vegas', 'sacramento', 'salt lake city', 'baltimore']
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

def extract_city_from_input(text: str) -> str:
    """Extract city name from buyer input for dynamic property adaptation."""
    text_lower = text.lower()

    # Common US cities to detect (expand as needed)
    cities = [
        'boston', 'new york', 'chicago', 'los angeles', 'seattle', 'austin',
        'denver', 'miami', 'atlanta', 'portland', 'phoenix', 'san diego',
        'dallas', 'houston', 'philadelphia', 'washington dc', 'detroit',
        'minneapolis', 'nashville', 'charlotte', 'raleigh', 'tampa',
        'orlando', 'las vegas', 'sacramento', 'salt lake city', 'baltimore'
    ]

    for city in cities:
        if city in text_lower:
            return city.title()

    return None

def adapt_properties_for_city(properties: list, target_city: str) -> list:
    """Adapt property addresses to target city for demo purposes."""
    # State mapping for cities
    city_states = {
        'Boston': 'MA', 'New York': 'NY', 'Chicago': 'IL', 'Los Angeles': 'CA',
        'Seattle': 'WA', 'Austin': 'TX', 'Denver': 'CO', 'Miami': 'FL',
        'Atlanta': 'GA', 'Portland': 'OR', 'Phoenix': 'AZ', 'San Diego': 'CA',
        'Dallas': 'TX', 'Houston': 'TX', 'Philadelphia': 'PA', 'Washington Dc': 'DC',
        'Detroit': 'MI', 'Minneapolis': 'MN', 'Nashville': 'TN', 'Charlotte': 'NC',
        'Raleigh': 'NC', 'Tampa': 'FL', 'Orlando': 'FL', 'Las Vegas': 'NV',
        'Sacramento': 'CA', 'Salt Lake City': 'UT', 'Baltimore': 'MD'
    }

    state = city_states.get(target_city, 'CA')
    adapted = []

    for prop in properties:
        adapted_prop = prop.copy()
        # Extract street number and name from original address
        original_addr = prop.get('address', '')
        parts = original_addr.split(',')
        if parts:
            street = parts[0].strip()
            # Create new address with target city
            adapted_prop['address'] = f"{street}, {target_city}, {state}"
            # Update neighborhood to reflect city
            adapted_prop['neighborhood'] = f"{target_city} - {prop.get('neighborhood', 'Downtown')}"
        adapted.append(adapted_prop)

    return adapted

def match_properties(preferences: dict, properties: list) -> list:
    """Match properties against buyer preferences."""
    # Check if buyer is looking for a different city
    preferred_locations = preferences.get('preferred_locations', [])
    target_city = None

    # Check if any preferred location is a major city not in our data
    # Bay Area cities are treated as local and won't trigger adaptation
    bay_area_cities = [
        # SF neighborhoods
        'pacific heights', 'marina', 'mission', 'soma', 'nopa', 'hayes valley',
        'russian hill', 'noe valley', 'richmond', 'sunset', 'san francisco',
        # South Bay
        'san jose', 'santa clara', 'sunnyvale', 'mountain view', 'palo alto',
        'cupertino', 'milpitas', 'saratoga', 'los gatos', 'campbell', 'morgan hill', 'gilroy',
        # East Bay
        'fremont', 'oakland', 'berkeley', 'hayward', 'concord', 'walnut creek',
        'pleasanton', 'livermore', 'newark', 'union city', 'alameda', 'san leandro',
        'san ramon', 'dublin',
        # Peninsula
        'san mateo', 'redwood city', 'daly city', 'south san francisco', 'menlo park',
        'foster city', 'burlingame', 'san bruno', 'pacifica',
        # North Bay
        'vallejo', 'fairfield', 'napa', 'sonoma', 'petaluma', 'santa rosa',
        'san rafael', 'novato', 'sausalito', 'tiburon'
    ]

    for loc in preferred_locations:
        loc_lower = loc.lower()
        if loc_lower not in bay_area_cities:
            # Check if it's a recognized city outside Bay Area
            if loc_lower in [c.lower() for c in ['Boston', 'New York', 'Chicago', 'Los Angeles',
                'Seattle', 'Austin', 'Denver', 'Miami', 'Atlanta', 'Portland', 'Phoenix',
                'San Diego', 'Dallas', 'Houston', 'Philadelphia', 'Washington DC', 'Detroit',
                'Minneapolis', 'Nashville', 'Charlotte', 'Raleigh', 'Tampa', 'Orlando',
                'Las Vegas', 'Sacramento', 'Salt Lake City', 'Baltimore']]:
                target_city = loc.title()
                break

    # Adapt properties if searching for different city
    if target_city:
        properties = adapt_properties_for_city(properties, target_city)
        logger.info(f"Adapted properties for city: {target_city}")

    matches = []

    for idx, prop in enumerate(properties):
        score = 10  # Base score for all properties
        reasons = []

        # Budget scoring with granularity
        max_budget = preferences.get('max_budget')
        if max_budget and 'price' in prop:
            price = prop['price']
            if price <= max_budget * 0.7:
                score += 30
                reasons.append("Well under budget")
            elif price <= max_budget * 0.9:
                score += 40
                reasons.append("Great value for budget")
            elif price <= max_budget:
                score += 35
                reasons.append("Within budget")
            elif price <= max_budget * 1.1:
                score += 15
                reasons.append("Slightly over budget")
            elif price <= max_budget * 1.2:
                score += 5
                reasons.append("Over budget but notable")

            # Bonus for price efficiency (lower price = small bonus)
            price_efficiency = max(0, (max_budget - price) / max_budget * 10)
            score += int(price_efficiency)
        elif not max_budget:
            score += 20  # No budget specified

        # Bedroom scoring with granularity
        min_bedrooms = preferences.get('min_bedrooms')
        if min_bedrooms and 'bedrooms' in prop:
            beds = prop['bedrooms']
            if beds == min_bedrooms:
                score += 20
                reasons.append(f"Exact {beds}BR match")
            elif beds == min_bedrooms + 1:
                score += 18
                reasons.append(f"{beds}BR - extra room")
            elif beds > min_bedrooms:
                score += 12
                reasons.append(f"{beds}BR meets requirement")

        # Location scoring
        if 'preferred_locations' in preferences and 'neighborhood' in prop:
            if prop['neighborhood'].lower() in [loc.lower() for loc in preferences.get('preferred_locations', [])]:
                score += 25
                reasons.append(f"In preferred area: {prop['neighborhood']}")

        # Feature scoring with individual weights
        if 'desired_features' in preferences and 'features' in prop:
            matching_features = set(f.lower() for f in preferences.get('desired_features', [])) & \
                              set(f.lower() for f in prop.get('features', []))
            if matching_features:
                # Variable points per feature (15-20 each)
                for i, feature in enumerate(matching_features):
                    score += 15 + (i % 6)  # Varies between 15-20
                reasons.append(f"Has: {', '.join(matching_features)}")

        # Sqft bonus (larger = small bonus)
        sqft = prop.get('sqft', 0)
        if sqft > 2000:
            score += 5
        elif sqft > 1500:
            score += 3

        # Small uniqueness factor based on property index to break ties
        score += (idx * 7) % 5  # Adds 0-4 points

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

def calculate_match_health(preferences: dict, matches: list) -> dict:
    """Calculate health metrics for recommendation quality."""
    if len(matches) < 2:
        return {'quality': 'insufficient_data', 'score': 0}

    health = {}

    # 1. Score Distribution Quality
    scores = [m['score'] for m in matches]
    top_score = scores[0]
    second_score = scores[1] if len(scores) > 1 else 0
    score_range = max(scores) - min(scores)

    health['score_distribution'] = {
        'top_score': top_score,
        'second_score': second_score,
        'score_gap': top_score - second_score,
        'score_range': score_range,
        'differentiation': 'good' if score_range > 20 else 'fair' if score_range > 10 else 'poor'
    }

    # 2. Preference Coverage
    total_prefs = 0
    matched_prefs = 0

    # Budget coverage
    if preferences.get('max_budget'):
        total_prefs += 1
        if any(m['property']['price'] <= preferences['max_budget'] for m in matches[:2]):
            matched_prefs += 1

    # Bedroom coverage
    if preferences.get('min_bedrooms'):
        total_prefs += 1
        if any(m['property'].get('bedrooms', 0) >= preferences['min_bedrooms'] for m in matches[:2]):
            matched_prefs += 1

    # Location coverage
    if preferences.get('preferred_locations'):
        total_prefs += 1
        pref_locs = [loc.lower() for loc in preferences['preferred_locations']]
        if any(m['property'].get('neighborhood', '').lower() in pref_locs for m in matches[:2]):
            matched_prefs += 1

    # Feature coverage
    if preferences.get('desired_features'):
        desired = set(f.lower() for f in preferences['desired_features'])
        for feature in desired:
            total_prefs += 1
            # Check if any top 2 match has this feature
            for m in matches[:2]:
                prop_features = set(f.lower() for f in m['property'].get('features', []))
                if feature in prop_features:
                    matched_prefs += 1
                    break

    coverage_pct = (matched_prefs / max(total_prefs, 1)) * 100
    health['preference_coverage'] = {
        'matched': matched_prefs,
        'total': total_prefs,
        'percentage': round(coverage_pct, 1)
    }

    # 3. Budget Efficiency
    budget = preferences.get('max_budget')
    if budget:
        top_prices = [m['property']['price'] for m in matches[:2]]
        avg_price = sum(top_prices) / len(top_prices)
        health['budget_efficiency'] = {
            'avg_price': int(avg_price),
            'utilization_pct': round((avg_price / budget) * 100, 1),
            'all_within_budget': all(p <= budget for p in top_prices),
            'best_savings': int(budget - min(top_prices))
        }

    # 4. Overall Quality Score (0-100)
    quality_score = 0

    # Score differentiation (0-30 pts)
    if score_range > 30:
        quality_score += 30
    elif score_range > 20:
        quality_score += 20
    elif score_range > 10:
        quality_score += 10

    # Preference coverage (0-40 pts)
    quality_score += int(coverage_pct * 0.4)

    # Budget efficiency (0-30 pts)
    if budget:
        if health['budget_efficiency']['all_within_budget']:
            quality_score += 20
            # Bonus for good utilization (70-100%)
            util = health['budget_efficiency']['utilization_pct']
            if 70 <= util <= 100:
                quality_score += 10
    else:
        quality_score += 15  # No budget = neutral

    # Quality label
    if quality_score >= 80:
        quality_label = 'excellent'
    elif quality_score >= 60:
        quality_label = 'good'
    elif quality_score >= 40:
        quality_label = 'fair'
    else:
        quality_label = 'poor'

    health['overall'] = {
        'score': quality_score,
        'quality': quality_label
    }

    return health

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
    """Health check endpoint with dependency verification"""
    checks = {
        'server': True,
        'properties_loaded': False,
        'api_keys_configured': False,
        'property_count': 0
    }

    # Check properties file loads correctly
    try:
        props = load_properties()
        checks['properties_loaded'] = len(props) > 0
        checks['property_count'] = len(props)
    except Exception as e:
        checks['properties_loaded'] = False
        logger.warning(f"Health check: properties failed to load - {e}")

    # Check API keys are configured
    checks['api_keys_configured'] = all([
        os.environ.get('SYNTHETIC_API_KEY'),
        os.environ.get('SENDGRID_API_KEY'),
        os.environ.get('SENDGRID_FROM_EMAIL')
    ])

    # Overall status
    all_healthy = all([
        checks['server'],
        checks['properties_loaded'],
        checks['api_keys_configured']
    ])

    return jsonify({
        'status': 'healthy' if all_healthy else 'degraded',
        'service': 'scout-ai',
        'checks': checks
    }), 200 if all_healthy else 503

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
            return jsonify(safe_fail('Buyer input is required', trace_id)), 400

        if not buyer_email:
            metrics['failed_emails'] += 1
            return jsonify(safe_fail('Buyer email is required', trace_id)), 400

        # Security: Validate and sanitize all inputs
        is_valid, error_msg, sanitized = validate_input(
            buyer_input, buyer_name, buyer_email, personalization_style
        )
        if not is_valid:
            metrics['failed_emails'] += 1
            logger.warning(f"[{trace_id}] Input validation failed: {error_msg}")
            return jsonify(safe_fail(error_msg, trace_id)), 400

        # Use sanitized inputs
        buyer_input = sanitized['buyer_input']
        buyer_name = sanitized['buyer_name']
        buyer_email = sanitized['buyer_email']
        personalization_style = sanitized['style']

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

        # Security: Policy check before LLM call
        allowed, reason = check_llm_policy(len(llm_input), 'preference_analysis')
        if not allowed:
            logger.warning(f"[{trace_id}] LLM policy check failed: {reason}")
            return jsonify(safe_fail(reason, trace_id)), 429

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

        # Security: Policy check before LLM call
        allowed, reason = check_llm_policy(len(prompt_content), 'email_generation')
        if not allowed:
            logger.warning(f"[{trace_id}] LLM policy check failed: {reason}")
            return jsonify(safe_fail(reason, trace_id)), 429

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

        # Security: Validate and sanitize output
        is_valid, email_content = validate_output(email_content)
        if not is_valid:
            logger.warning(f"[{trace_id}] Output validation failed")
            # Note: validate_output always returns True but sanitizes content

        add_trace_step(trace, 'validate_output', 0, details={'sanitized': True})

        # Step 6: Send email
        step_start = time.time()

        # Security: Policy check before SendGrid API call
        allowed, reason = check_sendgrid_policy(buyer_email, len(email_content))
        if not allowed:
            metrics['failed_emails'] += 1
            logger.warning(f"[{trace_id}] SendGrid policy check failed: {reason}")
            return jsonify(safe_fail(reason, trace_id)), 429

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

        # Calculate match health metrics
        match_health = calculate_match_health(preferences, matches)
        logger.info(f"[{trace_id}] Match quality: {match_health['overall']['quality']} (score: {match_health['overall']['score']})")

        return jsonify({
            'status': 'success',
            'buyer_name': buyer_name,
            'buyer_email': buyer_email,
            'properties_matched': len(matches),
            'properties_selected': [m['address'] for m in selected_matches],
            'steps': steps,
            'email_content': email_content,
            'match_health': match_health,
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
