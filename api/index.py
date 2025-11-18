from flask import Flask, render_template, request, jsonify
import sys
import os

# Add parent directory to path so we can import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import run_campaign

app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')

# Simple rate limiting
rate_limit_cache = {}

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

        # Basic rate limiting
        ip = request.remote_addr
        if ip in rate_limit_cache:
            rate_limit_cache[ip] += 1
            if rate_limit_cache[ip] > 20:
                return jsonify({'error': 'Rate limit exceeded'}), 429
        else:
            rate_limit_cache[ip] = 1

        # Run the agent
        result = run_campaign(buyer_name, buyer_input)

        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred generating the campaign'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'homeMatch-ai'})
