# HomeMatch AI Agent - Property matching and email generation
import anthropic
import json
import os
from datetime import datetime

# Initialize Anthropic client
client = anthropic.Anthropic()

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
    """
    Use Claude to extract structured preferences from buyer input.
    Returns a dict with: budget, bedrooms, location, features, etc.
    """
    prompt = load_prompt('analyzer')

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt.format(buyer_input=buyer_input)
            }
        ]
    )

    # Parse the JSON response
    response_text = response.content[0].text
    try:
        # Extract JSON from response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(response_text[start:end])
    except json.JSONDecodeError:
        pass

    return {}

def match_properties(preferences: dict, properties: list) -> list:
    """
    Match properties against buyer preferences.
    Returns list of matching properties sorted by relevance.
    """
    matches = []

    for prop in properties:
        score = 0
        reasons = []

        # Budget check
        if 'max_budget' in preferences and 'price' in prop:
            if prop['price'] <= preferences['max_budget']:
                score += 30
                reasons.append("Within budget")
            elif prop['price'] <= preferences['max_budget'] * 1.1:
                score += 15
                reasons.append("Slightly over budget but close")

        # Bedrooms check
        if 'min_bedrooms' in preferences and 'bedrooms' in prop:
            if prop['bedrooms'] >= preferences['min_bedrooms']:
                score += 20
                reasons.append(f"{prop['bedrooms']} bedrooms meets requirement")

        # Location check
        if 'preferred_locations' in preferences and 'neighborhood' in prop:
            if prop['neighborhood'].lower() in [loc.lower() for loc in preferences.get('preferred_locations', [])]:
                score += 25
                reasons.append(f"In preferred area: {prop['neighborhood']}")

        # Features check
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

    # Sort by score descending
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches[:5]  # Return top 5 matches

def generate_email(buyer_name: str, preferences: dict, matches: list) -> str:
    """
    Use Claude to generate a personalized email with property recommendations.
    """
    prompt = load_prompt('email_gen')

    # Format matches for the prompt
    matches_text = ""
    for i, match in enumerate(matches, 1):
        prop = match['property']
        matches_text += f"\n{i}. {prop.get('address', 'Address TBD')}\n"
        matches_text += f"   Price: ${prop.get('price', 0):,}\n"
        matches_text += f"   Bedrooms: {prop.get('bedrooms', 'N/A')} | Bathrooms: {prop.get('bathrooms', 'N/A')}\n"
        matches_text += f"   Neighborhood: {prop.get('neighborhood', 'N/A')}\n"
        matches_text += f"   Features: {', '.join(prop.get('features', []))}\n"
        matches_text += f"   Why it matches: {', '.join(match['match_reasons'])}\n"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
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

    return response.content[0].text

def run_campaign(buyer_name: str, buyer_input: str) -> dict:
    """
    Main agent workflow:
    1. Analyze buyer preferences
    2. Match against property listings
    3. Generate personalized email

    Returns dict with analysis, matches, and generated email.
    """
    try:
        # Step 1: Load properties
        properties = load_properties()

        # Step 2: Analyze buyer preferences
        preferences = analyze_buyer_preferences(buyer_input)

        # Step 3: Find matching properties
        matches = match_properties(preferences, properties)

        if not matches:
            return {
                'status': 'no_matches',
                'buyer_name': buyer_name,
                'preferences': preferences,
                'message': 'No properties found matching your criteria. Please adjust your preferences.'
            }

        # Step 4: Generate personalized email
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
