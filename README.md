# Scout - AI Marketing Agent for Realtors

**Autonomous AI-Powered Real Estate Email Campaigns**

Transform how real estate agents connect buyers with their dream homes. Scout autonomously analyzes buyer preferences, matches properties, generates personalized emails, and delivers them - all without human intervention.

---

## The Problem

Real estate agents spend **hours each week** on repetitive tasks:
- Manually reviewing buyer preferences
- Searching through property listings
- Writing personalized emails for each client
- Following up with multiple buyers

This manual process is:
- **Time-consuming** - 2-3 hours per buyer
- **Inconsistent** - Quality varies with agent workload
- **Not scalable** - Limited by human bandwidth

---

## The Solution: Scout

An **autonomous AI agent** that handles the entire buyer outreach workflow end-to-end.

### One Click. Multiple Buyers. Zero Manual Work.

```
Agent Start -> Analyze Preferences -> Match Properties -> Generate Email -> Send -> Repeat
```

The agent processes your entire buyer queue automatically, sending personalized property recommendations at configurable intervals.

---

## Agentic Workflows

### What Makes This an "Agent"?

Unlike simple automation, Scout exhibits true **agentic behavior**:

| Characteristic | How Scout Implements It |
|----------------|-------------------------|
| **Autonomy** | Runs independently after initial configuration |
| **Goal-Oriented** | Completes the full email campaign without intervention |
| **Decision Making** | Analyzes preferences, scores properties, selects best matches |
| **Multi-Step Reasoning** | Chains multiple AI calls (analyze -> match -> generate) |
| **Observable** | Real-time activity log shows agent "thinking" |

### The Agent Loop

```
+----------------------------------+
|       AGENT CONTROL LOOP         |
+----------------------------------+

  1. PERCEIVE
     - Load buyer preferences from queue

  2. REASON
     - LLM extracts structured preferences
     - Scoring algorithm ranks properties
     - Auto-select top 2 matches

  3. ACT
     - LLM generates personalized HTML email
     - SendGrid delivers to buyer

  4. LOOP
     - Wait configured interval
     - Process next buyer in queue
     - Repeat until max emails reached
```

---

## Key Features

### AI-Powered Preference Analysis
Natural language understanding extracts:
- Budget constraints
- Bedroom/bathroom requirements
- Preferred neighborhoods
- Must-have features
- Nice-to-have amenities

**Input:** *"Looking for a 3BR in Pacific Heights or Marina, budget $1.5M, must have garage and modern kitchen"*

**Output:** Structured JSON with extracted preferences

### Intelligent Property Matching
Proprietary scoring algorithm weighs:
- Budget fit (40 points)
- Location match (20 points)
- Bedroom requirements (15 points)
- Feature matching (25 points per feature)

### Personalized Email Generation
LLM generates HTML emails featuring:
- Personalized greeting
- Buyer's summarized preferences
- Top 2 property recommendations with images
- Why each property matches their criteria
- Professional call-to-action

### Autonomous Batch Processing
- Configure email interval (5 seconds for demo)
- Set maximum emails per run
- Select buyers from queue
- Start agent and observe
- Stop anytime if needed

### Real-Time Activity Log
Watch the agent "think" in real-time:
```
12:34:01  Agent started. Processing 3 buyers at 5s intervals
12:34:01  Processing John Smith...
12:34:02  Analyzing preferences for John Smith
12:34:03  Found 5 matching properties
12:34:03  Selected top 2 properties (scores: 95, 80)
12:34:05  Generating personalized email
12:34:08  Sending email to john@example.com
12:34:08  Email delivered to john@example.com
12:34:08  Waiting 5 seconds before next email...
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.13 + Flask |
| **LLM API** | Synthetic API (GLM-4.5) |
| **Email Delivery** | SendGrid |
| **Frontend** | Vanilla JS + CSS |
| **Deployment** | Vercel Serverless |
| **Data** | JSON (properties) + localStorage (buyers) |

---

## Demo Flow

### For Hackathon Judges

1. **Setup (30 seconds)**
   - Show pre-saved buyers with different preferences
   - Highlight the "Launch Email Campaign" panel

2. **Configuration (15 seconds)**
   - Set frequency to "Every 5 seconds (Demo)"

3. **Launch Agent (10 seconds)**
   - Click "Launch Email Campaign"
   - Point to the "Thinking..." status indicator

4. **Observe Autonomy (60 seconds)**
   - Watch the activity log populate in real-time
   - See buyer statuses change: Pending -> Processing -> Sent
   - Note the progress bar advancing
   - Highlight the agent's decision-making in the logs

5. **Results (15 seconds)**
   - Show "Campaign Complete"
   - Check email inbox for delivered messages

**Total demo time: ~2 minutes**

---

## Value Proposition

### For Real Estate Agents

| Metric | Before | After |
|--------|--------|-------|
| Time per buyer | 2-3 hours | 30 seconds |
| Buyers contacted/day | 3-5 | 50+ |
| Email personalization | Variable | Consistently high |
| Follow-up consistency | Often forgotten | Automated |

### ROI Calculation

- **Time saved:** 10 hours/week
- **Additional contacts:** 200+ buyers/month
- **Conversion improvement:** 2-3x with personalized outreach
- **Agent value:** Focus on high-value activities (showings, negotiations)

---

## Future Roadmap

### Phase 2: Enhanced Intelligence
- [ ] Property preference learning from buyer feedback
- [ ] Optimal send-time prediction
- [ ] A/B testing email templates
- [ ] Multi-language support

### Phase 3: Full Automation
- [ ] New listing alerts to matching buyers
- [ ] Automated follow-up sequences
- [ ] CRM integration (Salesforce, HubSpot)
- [ ] MLS data feed integration

### Phase 4: Multi-Agent System
- [ ] Buyer qualification agent
- [ ] Showing scheduler agent
- [ ] Market analysis agent
- [ ] Negotiation assistant agent

---

## Getting Started

### Prerequisites
- Python 3.13+
- SendGrid API key
- Synthetic API key

### Installation

```bash
# Clone repository
git clone https://github.com/ellyphant/HomeMatchAI-Agent.git
cd HomeMatchAI-Agent

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.sample .env
# Edit .env with your API keys

# Run locally
python3 app.py
# Then open http://localhost:5000 in your browser
```

### Environment Variables

```env
SYNTHETIC_API_KEY=your_synthetic_api_key
SENDGRID_API_KEY=your_sendgrid_api_key
SENDGRID_FROM_EMAIL=your_verified_sender@domain.com
```

### Deploy to Vercel

```bash
vercel --prod
```

---

## Architecture

```
HomeMatchAI-Agent/
├── app.py                    # Flask app + all API routes
├── api/index.py              # Vercel serverless entry
├── data/
│   └── properties.json       # 15 SF & San Jose property listings
├── prompts/
│   ├── analyzer.txt          # Preference extraction prompt
│   └── email_gen.txt         # Email generation prompt
├── templates/
│   └── index.html            # Single-page application
├── requirements.txt          # Python dependencies
└── vercel.json               # Deployment config
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve web application |
| `/generate` | POST | Analyze preferences + match properties |
| `/generate-email` | POST | Generate email for selected properties |
| `/send-email` | POST | Send email via SendGrid |
| `/run-agent-cycle` | POST | **Full autonomous agent cycle** |
| `/health` | GET | Health check |

---

## Why We'll Win

### Technical Excellence
- Clean, modular architecture
- Real LLM integration (not mocked)
- Production-ready email delivery
- Deployable to Vercel in one command

### True Agentic Behavior
- Autonomous multi-step execution
- Observable decision-making
- Configurable behavior
- Graceful error handling

### Real-World Value
- Solves actual pain point for real estate agents
- Measurable ROI (10+ hours saved weekly)
- Clear path to monetization
- Scalable to other industries

### Demo Impact
- Visual "wow factor" with real-time activity log
- Actual emails delivered during demo
- Puppy mascot adds personality
- Easy to understand in 2 minutes

---

**Scout** - *Because your buyers deserve personalized attention, and you deserve your time back.*
