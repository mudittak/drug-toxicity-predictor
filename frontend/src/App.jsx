/**
 * App.jsx - Root component
 * Manages tab switching, API calls, and server health status
 */

import { useState, useEffect } from 'react'
import { FlaskConical, Layers, Wifi, WifiOff } from 'lucide-react'
import SMILESInput   from './components/SMILESInput'
import ResultCard    from './components/ResultCard'
import BatchAnalyzer from './components/BatchAnalyzer'
import { predictToxicity, checkHealth } from './services/api'

export default function App() {
  const [tab, setTab]         = useState('single')
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [health, setHealth]   = useState(null)

  // Check backend health on page load
  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth({ status: 'error' }))
  }, [])

  const handlePredict = async (smiles, name) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await predictToxicity(smiles, name)
      setResult(data)
    } catch {
      setError('Cannot connect to backend. Make sure the server is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  const serverOk = health?.status === 'ok'

  return (
    <div className="min-h-screen bg-bg grid-bg">

      {/* Top Navigation Bar */}
      <div className="border-b border-border bg-surface/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 py-3 flex items-center justify-between">

          {/* Brand */}
          <div className="flex items-center gap-2">
            <FlaskConical size={18} className="text-accent" />
            <span className="font-mono font-bold text-white text-sm">ToxPredict</span>
            <span className="text-muted text-xs font-mono hidden sm:block">/ Drug Toxicity ML</span>
          </div>

          <div className="flex items-center gap-4">
            {/* Server status */}
            {health && (
              <div className={`flex items-center gap-1.5 text-xs font-mono ${serverOk ? 'text-safe' : 'text-danger'}`}>
                {serverOk ? <Wifi size={12} /> : <WifiOff size={12} />}
                {serverOk
                  ? (health.model_trained ? 'MODEL READY' : 'TRAIN MODEL')
                  : 'SERVER DOWN'}
              </div>
            )}

            {/* Tab switcher */}
            <div className="flex bg-card border border-border rounded-lg p-0.5">
              {[
                { id: 'single', icon: FlaskConical, label: 'Single' },
                { id: 'batch',  icon: Layers,       label: 'Batch'  },
              ].map(t => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all ${tab === t.id ? 'bg-accent text-bg font-bold' : 'text-muted hover:text-white'}`}
                >
                  <t.icon size={12} />{t.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-6 py-12">
        {tab === 'single' ? (
          <>
            <SMILESInput onPredict={handlePredict} loading={loading} />
            {error && (
              <div className="w-full max-w-3xl mx-auto mt-6 bg-danger/10 border border-danger/30 rounded-xl px-5 py-4">
                <p className="text-danger text-sm font-mono text-center">{error}</p>
              </div>
            )}
            <ResultCard result={result} />
          </>
        ) : (
          <BatchAnalyzer />
        )}

        {/* Warning if model not trained */}
        {!health?.model_trained && health?.status === 'ok' && (
          <div className="w-full max-w-3xl mx-auto mt-8 bg-warn/10 border border-warn/30 rounded-xl px-5 py-4">
            <p className="text-warn text-xs font-mono text-center">
              Model not trained yet. Run: cd backend/ml_model then python train_model.py
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-16 py-6">
        <div className="max-w-4xl mx-auto px-6 flex items-center justify-between">
          <p className="text-muted text-xs font-mono">ToxPredict · CodeCure AI Hackathon 2026</p>
          <p className="text-muted text-xs font-mono">Random Forest · RDKit · Tox21</p>
        </div>
      </footer>
    </div>
  )
}