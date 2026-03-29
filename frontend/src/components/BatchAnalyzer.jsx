import { useState } from 'react'
import { Layers, Plus, Trash2, Zap, Download } from 'lucide-react'
import { predictBatch } from '../services/api'

const RISK_COLORS = {
  HIGH:     'text-danger bg-danger/10 border-danger/30',
  MODERATE: 'text-warn bg-warn/10 border-warn/30',
  LOW:      'text-yellow-400 bg-yellow-400/10 border-yellow-400/30',
  MINIMAL:  'text-safe bg-safe/10 border-safe/30',
}

export default function BatchAnalyzer() {
  const [rows, setRows] = useState([
    { name: 'Aspirin',      smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
    { name: 'Nitrobenzene', smiles: 'c1ccc(cc1)N(=O)=O' },
    { name: 'Ethanol',      smiles: 'CCO' },
  ])
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const addRow    = () => setRows(r => [...r, { name: '', smiles: '' }])
  const removeRow = (i) => setRows(r => r.filter((_, idx) => idx !== i))
  const updateRow = (i, field, val) =>
    setRows(r => r.map((row, idx) => idx === i ? { ...row, [field]: val } : row))

  const handleBatch = async () => {
    const valid = rows.filter(r => r.smiles.trim())
    if (!valid.length) return
    setLoading(true)
    setError(null)
    try {
      const data = await predictBatch(
        valid.map(r => ({ smiles: r.smiles.trim(), compound_name: r.name.trim() || undefined }))
      )
      setResults(data)
    } catch {
      setError('Backend not reachable. Make sure the server is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  const downloadCSV = () => {
    if (!results) return
    const headers = ['Compound', 'SMILES', 'Prediction', 'Toxic %', 'Risk Level', 'MW', 'LogP']
    const csvRows = results.results.map(r => [
      r.compound_name || '',
      r.smiles || '',
      r.prediction || 'ERROR',
      r.toxic_probability || '',
      r.risk_level || '',
      r.molecular_properties?.molecular_weight || '',
      r.molecular_properties?.logP || '',
    ])
    const csv  = [headers, ...csvRows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url
    a.download = 'toxicity_predictions.csv'
    a.click()
  }

  return (
    <div className="w-full max-w-3xl mx-auto animate-fade-in">
      <div className="bg-card border border-border rounded-2xl p-6">

        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-2">
            <Layers size={16} className="text-accent" />
            <h2 className="text-sm font-mono text-accent uppercase tracking-widest">Batch Analysis</h2>
          </div>
          <button onClick={addRow}
            className="flex items-center gap-1.5 text-xs font-mono text-muted hover:text-accent border border-border hover:border-accent/40 px-3 py-1.5 rounded-lg transition-all">
            <Plus size={12} /> ADD ROW
          </button>
        </div>

        <div className="space-y-2 mb-5">
          <div className="grid grid-cols-5 gap-2 text-xs font-mono text-muted px-1 pb-1">
            <span className="col-span-1">NAME</span>
            <span className="col-span-3">SMILES</span>
            <span></span>
          </div>
          {rows.map((row, i) => (
            <div key={i} className="grid grid-cols-5 gap-2">
              <input value={row.name} onChange={e => updateRow(i, 'name', e.target.value)}
                placeholder="Name"
                className="col-span-1 bg-surface border border-border rounded-lg px-3 py-2 text-xs text-white placeholder:text-muted/40 focus:outline-none focus:border-accent/60 transition-colors" />
              <input value={row.smiles} onChange={e => updateRow(i, 'smiles', e.target.value)}
                placeholder="SMILES string"
                className="col-span-3 bg-surface border border-border rounded-lg px-3 py-2 text-xs font-mono text-white placeholder:text-muted/40 focus:outline-none focus:border-accent/60 transition-colors" />
              <button onClick={() => removeRow(i)} disabled={rows.length === 1}
                className="flex items-center justify-center text-muted hover:text-danger border border-border hover:border-danger/30 rounded-lg transition-all disabled:opacity-30">
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        <button onClick={handleBatch}
          disabled={loading || !rows.some(r => r.smiles.trim())}
          className="w-full flex items-center justify-center gap-2 bg-accent text-bg font-mono font-bold py-3 rounded-xl hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed transition-all text-sm">
          {loading
            ? <><div className="w-4 h-4 border-2 border-bg/40 border-t-bg rounded-full animate-spin" />ANALYZING...</>
            : <><Zap size={16} />ANALYZE {rows.filter(r => r.smiles.trim()).length} COMPOUNDS</>
          }
        </button>
        {error && <p className="text-danger text-xs font-mono text-center mt-3">{error}</p>}
      </div>

      {results && (
        <div className="mt-4 bg-card border border-border rounded-2xl p-6 animate-slide-up">

          <div className="grid grid-cols-3 gap-3 mb-5">
            {[
              { label: 'TOTAL', val: results.total,       color: 'text-white'  },
              { label: 'TOXIC', val: results.toxic_count, color: 'text-danger' },
              { label: 'SAFE',  val: results.safe_count,  color: 'text-safe'   },
            ].map(s => (
              <div key={s.label} className="bg-surface rounded-xl p-3 text-center">
                <p className={'text-2xl font-mono font-bold ' + s.color}>{s.val}</p>
                <p className="text-muted text-xs font-mono">{s.label}</p>
              </div>
            ))}
          </div>

          <div className="space-y-2">
            {results.results.map((r, i) => (
              <div key={i} className="flex items-center justify-between bg-surface rounded-xl px-4 py-3 gap-3">
                <div className="min-w-0 flex-1">
                  <p className="text-white text-sm font-medium truncate">{r.compound_name || 'Compound ' + (i + 1)}</p>
                  <p className="text-muted text-xs font-mono truncate">{r.smiles}</p>
                </div>
                <div className="flex items-center gap-3 flex-shrink-0">
                  <span className="text-white text-sm font-mono">{r.toxic_probability}%</span>
                  <span className={'text-xs font-mono font-bold px-2 py-1 rounded-lg border ' + (RISK_COLORS[r.risk_level] || 'text-muted')}>
                    {r.risk_level || 'N/A'}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <button onClick={downloadCSV}
            className="mt-4 w-full flex items-center justify-center gap-2 border border-border text-muted hover:text-accent hover:border-accent/40 font-mono text-xs py-2.5 rounded-xl transition-all">
            <Download size={14} /> DOWNLOAD CSV
          </button>
        </div>
      )}
    </div>
  )
}