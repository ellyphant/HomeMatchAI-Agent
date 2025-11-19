from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
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
            return {}

        response_text = response.choices[0].message.content
        if not response_text:
            print("Empty response content")
            return {}

        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(response_text[start:end])

        print(f"No JSON found in response: {response_text[:200]}")
        return {}

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return {}
    except Exception as e:
        print(f"Error in analyze_buyer_preferences: {e}")
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

    return response.choices[0].message.content or "Email generation failed"

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
    return jsonify({'status': 'healthy', 'service': 'homeMatch-ai'})

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
        if is_html:
            message = Mail(
                from_email=os.environ.get('SENDGRID_FROM_EMAIL', 'noreply@homematch.ai'),
                to_emails=to_email,
                subject=subject,
                html_content=email_content
            )
        else:
            message = Mail(
                from_email=os.environ.get('SENDGRID_FROM_EMAIL', 'noreply@homematch.ai'),
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
