# Scout Operations Runbook

## Quick Reference

### Health Check
```bash
curl http://localhost:5000/health
# Expected: {"status": "healthy", "service": "scout-ai"}
```

### View Metrics
```bash
curl http://localhost:5000/metrics | jq
```

### View Logs (Local)
```bash
# Logs are printed to stdout with format:
# YYYY-MM-DD HH:MM:SS [LEVEL] message
tail -f server.log
```

---

## Debugging Common Issues

### 1. LLM API Errors

**Symptoms:**
- `llm_errors` metric increasing
- Error logs: "Error in analyze_buyer_preferences"
- Preferences returning empty

**Debug Steps:**
```bash
# Check metrics for LLM errors
curl http://localhost:5000/metrics | jq '.metrics.llm_errors'

# Check recent traces for failed LLM calls
curl http://localhost:5000/metrics | jq '.recent_traces[-1].llm_calls'

# Verify API key is set
echo $SYNTHETIC_API_KEY | head -c 10
```

**Root Causes:**
- Invalid/expired API key
- Synthetic API rate limiting
- Network connectivity issues

**Resolution:**
1. Verify API key in `.env`
2. Check Synthetic API status page
3. Review input text for malformed content

### 2. Email Delivery Failures

**Symptoms:**
- `failed_emails` metric increasing
- Error: "SendGrid API error"
- Emails not received

**Debug Steps:**
```bash
# Check success rate
curl http://localhost:5000/metrics | jq '.metrics.success_rate'

# Check SendGrid dashboard for bounces/blocks
# https://app.sendgrid.com/email_activity
```

**Root Causes:**
- Invalid recipient email
- SendGrid API key invalid
- Sender not verified
- Rate limiting

**Resolution:**
1. Verify SendGrid API key in `.env`
2. Check sender email is verified in SendGrid
3. Review SendGrid activity for specific errors

### 3. Slow Response Times

**Symptoms:**
- `avg_latency_ms` > 10000
- Users reporting timeouts
- Progress bar stuck

**Debug Steps:**
```bash
# Check average latency
curl http://localhost:5000/metrics | jq '.metrics.avg_latency_ms'

# Check step-by-step timing in recent trace
curl http://localhost:5000/metrics | jq '.recent_traces[-1].steps'
```

**Root Causes:**
- LLM API latency
- Large email content generation
- SendGrid API slowness

**Resolution:**
1. Check which step is slowest in traces
2. Consider reducing max_tokens for LLM calls
3. Implement caching for repeated queries

### 4. LLM Fallback Triggered

**Symptoms:**
- `llm_fallbacks` metric increasing
- Preferences missing expected fields
- Warning logs: "LLM fallback triggered"

**Debug Steps:**
```bash
# Check fallback count
curl http://localhost:5000/metrics | jq '.metrics.llm_fallbacks'

# Review LLM input/output in trace
curl http://localhost:5000/metrics | jq '.recent_traces[-1].llm_calls[0]'
```

**Root Causes:**
- Malformed buyer input
- LLM returning non-JSON response
- Model changes in Synthetic API

**Resolution:**
1. Review buyer input text
2. Check LLM output preview in trace
3. Update prompt template if needed

---

## Reproducing Issues

### Reproduce LLM Error
```bash
curl -X POST http://localhost:5000/run-agent-cycle \
  -H "Content-Type: application/json" \
  -d '{
    "buyer_name": "Test User",
    "buyer_email": "test@example.com",
    "buyer_input": ""
  }'
# Should fail validation
```

### Reproduce with Trace
```bash
# Run agent cycle and capture trace_id
RESPONSE=$(curl -s -X POST http://localhost:5000/run-agent-cycle \
  -H "Content-Type: application/json" \
  -d '{
    "buyer_name": "Test User",
    "buyer_email": "test@example.com",
    "buyer_input": "3BR in Marina, budget 1.5M"
  }')

# Get trace_id
echo $RESPONSE | jq '.trace_id'

# View full trace in metrics
curl http://localhost:5000/metrics | jq '.recent_traces[-1]'
```

---

## Rollback Procedures

### Rollback Deployment (Vercel)
```bash
# List deployments
vercel ls

# Rollback to previous deployment
vercel rollback <deployment-url>
```

### Rollback Code Changes
```bash
# View recent commits
git log --oneline -10

# Revert to specific commit
git revert <commit-hash>
git push

# Or hard reset (use with caution)
git reset --hard <commit-hash>
git push --force
```

### Disable Agent (Emergency)
1. Comment out `/run-agent-cycle` route in app.py
2. Deploy immediately
3. Users will see error, but no emails sent

---

## SLIs and SLOs

| Metric | SLI | SLO Target |
|--------|-----|------------|
| Success Rate | `successful_emails / total_requests` | > 95% |
| Latency (p50) | `avg_latency_ms` | < 5000ms |
| LLM Error Rate | `llm_errors / llm_calls` | < 5% |
| Fallback Rate | `llm_fallbacks / llm_calls` | < 10% |

### Check SLO Compliance
```bash
curl http://localhost:5000/metrics | jq '{
  success_rate: .metrics.success_rate,
  avg_latency: .metrics.avg_latency_ms,
  llm_calls: .metrics.llm_calls,
  llm_errors: .metrics.llm_errors
}'
```

---

## Trace Format Reference

Each agent cycle generates a trace with:

```json
{
  "trace_id": "abc12345",
  "buyer_name": "John Smith",
  "start_time": "2024-11-19T12:00:00Z",
  "end_time": "2024-11-19T12:00:05Z",
  "total_duration_ms": 5000,
  "status": "success",
  "steps": [
    {
      "step": "load_properties",
      "duration_ms": 10,
      "status": "success",
      "details": {"count": 15}
    },
    {
      "step": "analyze_preferences",
      "duration_ms": 1500,
      "status": "success",
      "details": {"preferences_extracted": 4}
    }
  ],
  "llm_calls": [
    {
      "type": "preference_analysis",
      "input_preview": "Buyer input: Looking for...",
      "output_preview": "{\"max_budget\": 1500000...",
      "duration_ms": 1500
    }
  ]
}
```

---

## Contact

- **On-call**: Check #scout-oncall Slack channel
- **Escalation**: @elly for production issues

---

*Last updated: November 2024*
