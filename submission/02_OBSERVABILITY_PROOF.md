# Observability Proof

## Live Endpoints

**Production Metrics**: https://homematch-agent.vercel.app/metrics

**Production Health**: https://homematch-agent.vercel.app/health

---

## Observability Features

### 1. End-to-End Tracing

Every agent cycle generates a unique `trace_id` that tracks:
- Step timing (load properties, analyze preferences, match, generate email, send)
- LLM call inputs/outputs (scrubbed of secrets)
- Success/failure status
- Total duration

### 2. Metrics Collection

The `/metrics` endpoint exposes:
- `total_requests` - Total agent cycles
- `successful_emails` - Emails sent successfully
- `failed_emails` - Failed attempts
- `success_rate` - Percentage success
- `llm_calls` - Total LLM API calls
- `llm_errors` - LLM failures
- `llm_fallbacks` - Fallback parser usage
- `avg_latency_ms` - Average cycle time

### 3. Secrets Scrubbing

All logs automatically scrub:
- Email addresses -> `[EMAIL]`
- API keys -> `[REDACTED]`

---

## Sample Metrics Output

```json
{
  "metrics": {
    "total_requests": 5,
    "successful_emails": 4,
    "failed_emails": 1,
    "success_rate": 80.0,
    "llm_calls": 10,
    "llm_errors": 0,
    "llm_fallbacks": 1,
    "avg_latency_ms": 4523.45
  },
  "recent_traces": [
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
          "status": "success"
        },
        {
          "step": "analyze_preferences",
          "duration_ms": 1500,
          "status": "success"
        },
        {
          "step": "match_properties",
          "duration_ms": 5,
          "status": "success"
        },
        {
          "step": "generate_email",
          "duration_ms": 2500,
          "status": "success"
        },
        {
          "step": "send_email",
          "duration_ms": 985,
          "status": "success"
        }
      ],
      "llm_calls": [
        {
          "type": "preference_analysis",
          "input_preview": "Buyer input: Looking for 3BR in Marina...",
          "output_preview": "{\"max_budget\": 1500000...",
          "duration_ms": 1500
        },
        {
          "type": "email_generation",
          "input_preview": "Generate HTML email for...",
          "output_preview": "<!DOCTYPE html>...",
          "duration_ms": 2500
        }
      ]
    }
  ],
  "timestamp": "2024-11-19T12:05:00Z"
}
```

---

## SLIs and SLOs

| Metric | SLI | SLO Target |
|--------|-----|------------|
| Success Rate | `successful_emails / total_requests` | > 95% |
| Latency (p50) | `avg_latency_ms` | < 5000ms |
| LLM Error Rate | `llm_errors / llm_calls` | < 5% |
| Fallback Rate | `llm_fallbacks / llm_calls` | < 10% |

---

## Local Verification

```bash
# Health check
curl http://localhost:5000/health

# View metrics
curl http://localhost:5000/metrics | jq

# Check SLO compliance
curl http://localhost:5000/metrics | jq '{
  success_rate: .metrics.success_rate,
  avg_latency: .metrics.avg_latency_ms,
  llm_errors: .metrics.llm_errors
}'
```
