import { generateObject } from 'ai'

interface CampaignRequest {
  agentName: string
  buyerProfile: string
}

interface EmailCampaign {
  subject: string
  preview: string
  body: string
  callToAction: string
}

export async function POST(request: Request): Promise<Response> {
  try {
    const body: CampaignRequest = await request.json()

    if (!body.agentName || !body.buyerProfile) {
      return Response.json(
        { error: 'Missing required fields' },
        { status: 400 }
      )
    }

    const prompt = `You are an expert real estate marketing specialist. Create a compelling email marketing campaign for a real estate agent.

Real Estate Agent: ${body.agentName}
Buyer Profile: ${body.buyerProfile}

Generate a personalized email campaign that:
1. Opens with a personalized, engaging subject line
2. Includes a preview that compels them to open the email
3. Has a warm, professional email body addressing their specific interests
4. Includes a strong call-to-action
5. Should be about 150-200 words in the body

Make it feel personal, not generic. Reference specific details from their profile.`

    const { object } = await generateObject({
      model: 'openai/gpt-4o-mini',
      schema: {
        type: 'object',
        properties: {
          subject: {
            type: 'string',
            description: 'Email subject line (max 60 characters)'
          },
          preview: {
            type: 'string',
            description: 'Email preview text (max 100 characters)'
          },
          body: {
            type: 'string',
            description: 'Main email body content'
          },
          callToAction: {
            type: 'string',
            description: 'Clear call-to-action text'
          }
        },
        required: ['subject', 'preview', 'body', 'callToAction']
      },
      prompt,
      system: 'You are a professional real estate marketing expert creating personalized email campaigns.'
    })

    return Response.json(object as EmailCampaign)
  } catch (error) {
    console.error('Campaign generation error:', error)
    return Response.json(
      { error: 'Failed to generate campaign' },
      { status: 500 }
    )
  }
}
