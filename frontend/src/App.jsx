import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_BASE = 'http://localhost:8000'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [health, setHealth] = useState('checking')
  const messagesEndRef = useRef(null)

  useEffect(() => {
    checkHealth()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`)
      if (res.ok) setHealth('healthy')
      else setHealth('unhealthy')
    } catch {
      setHealth('unhealthy')
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = { role: 'user', content: input.trim() }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [...messages, userMessage] })
      })

      if (!res.ok) {
        const error = await res.json()
        throw new Error(error.detail || 'Request failed')
      }

      const data = await res.json()
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.reply,
        recommendations: data.recommendations,
        endOfConversation: data.end_of_conversation
      }])
    } catch (err) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Error: ${err.message}`,
        error: true
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>SHL Assessment Recommender</h1>
        <div className={`health-indicator ${health}`}>
          {health === 'healthy' ? '🟢 API Connected' : '🔴 API Disconnected'}
        </div>
      </header>

      <main className="chat-container">
        <div className="messages" role="log" aria-live="polite">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role} ${msg.error ? 'error' : ''}`}>
              <div className="message-content">
                <p>{msg.content}</p>
                {msg.recommendations && msg.recommendations.length > 0 && (
                  <div className="recommendations">
                    <h4>Recommended Assessments:</h4>
                    <ul>
                      {msg.recommendations.map((rec, i) => (
                        <li key={i}>
                          <a href={rec.url} target="_blank" rel="noopener noreferrer">
                            {rec.name}
                          </a>
                          <span className="test-type">Type: {rec.test_type}</span>
                        </li>
                      ))}
                    </ul>
                    {msg.endOfConversation && (
                      <p className="end-notice">✓ Conversation complete</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message assistant loading">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span><span></span><span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form className="input-form" onSubmit={(e) => { e.preventDefault(); sendMessage() }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe your hiring needs... (Shift+Enter for new line)"
            disabled={isLoading}
            rows={3}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </form>
      </main>
    </div>
  )
}

export default App