from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
from datetime import datetime

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI-compatible client for Synthetic API
client = OpenAI(
    api_key=os.environ.get("SYNTHETIC_API_KEY"),
    base_url="https://api.synthetic.new/openai/v1"
)

# Simple rate limiting (in-memory)
rate_limit_cache = {}

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

    response_text = response.choices[0].message.content
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(response_text[start:end])
    except json.JSONDecodeError:
        pass

    return {}

def match_properties(preferences: dict, properties: list) -> list:
    """Match properties against buyer preferences."""
    matches = []

    for prop in properties:
        score = 0
        reasons = []

        max_budget = preferences.get('max_budget')
        if max_budget and 'price' in prop:
            if prop['price'] <= max_budget:
                score += 30
                reasons.append("Within budget")
            elif prop['price'] <= max_budget * 1.1:
                score += 15
                reasons.append("Slightly over budget but close")

        min_bedrooms = preferences.get('min_bedrooms')
        if min_bedrooms and 'bedrooms' in prop:
            if prop['bedrooms'] >= min_bedrooms:
                score += 20
                reasons.append(f"{prop['bedrooms']} bedrooms meets requirement")

        if 'preferred_locations' in preferences and 'neighborhood' in prop:
            if prop['neighborhood'].lower() in [loc.lower() for loc in preferences.get('preferred_locations', [])]:
                score += 25
                reasons.append(f"In preferred area: {prop['neighborhood']}")

        if 'desired_features' in preferences and 'features' in prop:
            matching_features = set(f.lower() for f in preferences.get('desired_features', [])) & \
                              set(f.lower() for f in prop.get('features', []))
            if matching_features:
                score += len(matching_features) * 5
                reasons.append(f"Has: {', '.join(matching_features)}")

        if score > 0:
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

    return response.choices[0].message.content

def run_campaign(buyer_name: str, buyer_input: str) -> dict:
    """Main agent workflow."""
    try:
        properties = load_properties()
        preferences = analyze_buyer_preferences(buyer_input)
        matches = match_properties(preferences, properties)

        if not matches:
            return {
                'status': 'no_matches',
                'buyer_name': buyer_name,
                'preferences': preferences,
                'message': 'No properties found matching your criteria.'
            }

        email = generate_email(buyer_name, preferences, matches)

        return {
            'status': 'success',
            'buyer_name': buyer_name,
            'preferences': preferences,
            'matches': [
                {
                    'address': m['property'].get('address'),
                    'price': m['property'].get('price'),
                    'score': m['score'],
                    'reasons': m['match_reasons']
                }
                for m in matches
            ],
            'email': email,
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

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'homeMatch-ai'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
