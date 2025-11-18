'use client'

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Spinner } from '@/components/ui/spinner'
import { Mail, Home } from 'lucide-react'
import { EmailPreview } from '@/components/email-preview'

interface EmailCampaign {
  subject: string
  preview: string
  body: string
  callToAction: string
}

export default function Page() {
  const [agentName, setAgentName] = useState('')
  const [buyerProfile, setBuyerProfile] = useState('')
  const [loading, setLoading] = useState(false)
  const [campaign, setCampaign] = useState<EmailCampaign | null>(null)
  const [error, setError] = useState('')

  const handleGenerateCampaign = async () => {
    if (!agentName.trim() || !buyerProfile.trim()) {
      setError('Please fill in all fields')
      return
    }

    setLoading(true)
    setError('')
    setCampaign(null)

    try {
      const response = await fetch('/api/generate-campaign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agentName, buyerProfile }),
      })

      if (!response.ok) {
        throw new Error('Failed to generate campaign')
      }

      const data = await response.json()
      setCampaign(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="mb-12 text-center">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Home className="w-8 h-8 text-primary" />
            <h1 className="text-4xl font-bold tracking-tight">HomeMatch Agent</h1>
          </div>
          <p className="text-xl text-muted-foreground">AI-Powered Email Campaigns for Real Estate Agents</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <Card className="p-8">
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold">Create Campaign</h2>
              
              <div className="space-y-2">
                <Label htmlFor="agent">Your Name</Label>
                <Input
                  id="agent"
                  placeholder="e.g., Sarah Martinez"
                  value={agentName}
                  onChange={(e) => setAgentName(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="buyer">Buyer Profile & Interests</Label>
                <Textarea
                  id="buyer"
                  placeholder="e.g., First-time home buyer interested in eco-friendly homes in suburban areas, budget $350k-$450k, needs home near good schools..."
                  value={buyerProfile}
                  onChange={(e) => setBuyerProfile(e.target.value)}
                  rows={6}
                />
              </div>

              {error && (
                <div className="p-3 bg-destructive/10 text-destructive rounded-md text-sm">
                  {error}
                </div>
              )}

              <Button
                onClick={handleGenerateCampaign}
                disabled={loading}
                size="lg"
                className="w-full"
              >
                {loading ? (
                  <>
                    <Spinner className="mr-2 h-4 w-4" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Mail className="mr-2 h-4 w-4" />
                    Generate Campaign
                  </>
                )}
              </Button>
            </div>
          </Card>

          {/* Preview Section */}
          <div>
            {campaign ? (
              <EmailPreview campaign={campaign} agentName={agentName} />
            ) : (
              <Card className="p-8 h-full flex items-center justify-center bg-muted">
                <div className="text-center">
                  <Mail className="w-12 h-12 text-muted-foreground mb-4 mx-auto" />
                  <p className="text-muted-foreground">Your email campaign will appear here</p>
                </div>
              </Card>
            )}
          </div>
        </div>
      </div>
    </main>
  )
}
