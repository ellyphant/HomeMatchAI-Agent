'use client'

import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Copy, Download } from 'lucide-react'
import { useState } from 'react'

interface EmailPreviewProps {
  campaign: {
    subject: string
    preview: string
    body: string
    callToAction: string
  }
  agentName: string
}

export function EmailPreview({ campaign, agentName }: EmailPreviewProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    const emailContent = `Subject: ${campaign.subject}\n\n${campaign.body}\n\n${campaign.callToAction}\n\n${agentName}`
    navigator.clipboard.writeText(emailContent)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleDownload = () => {
    const emailContent = `Subject: ${campaign.subject}\n\n${campaign.body}\n\n${campaign.callToAction}\n\n${agentName}`
    const element = document.createElement('a')
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(emailContent))
    element.setAttribute('download', 'email-campaign.txt')
    element.style.display = 'none'
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  return (
    <Card className="p-8 h-full flex flex-col">
      <div className="flex-1 overflow-y-auto mb-6">
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-medium text-muted-foreground mb-2">Subject Line</h3>
            <p className="text-lg font-semibold">{campaign.subject}</p>
          </div>

          <div>
            <h3 className="text-sm font-medium text-muted-foreground mb-2">Preview</h3>
            <p className="text-sm text-foreground/80">{campaign.preview}</p>
          </div>

          <div className="my-6 p-4 bg-muted rounded-lg">
            <h3 className="text-sm font-medium text-muted-foreground mb-3">Email Body</h3>
            <div className="prose prose-sm max-w-none text-sm whitespace-pre-wrap">
              {campaign.body}
            </div>
          </div>

          <div className="p-4 bg-primary/5 rounded-lg border border-primary/20">
            <p className="text-sm font-semibold text-primary">{campaign.callToAction}</p>
          </div>
        </div>
      </div>

      <div className="flex gap-3">
        <Button
          variant="outline"
          size="sm"
          onClick={handleCopy}
          className="flex-1"
        >
          <Copy className="w-4 h-4 mr-2" />
          {copied ? 'Copied!' : 'Copy'}
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleDownload}
          className="flex-1"
        >
          <Download className="w-4 h-4 mr-2" />
          Download
        </Button>
      </div>
    </Card>
  )
}
