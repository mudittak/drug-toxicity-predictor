import { useState } from 'react'
import { Search, Zap, ChevronDown, ChevronUp, FlaskConical } from 'lucide-react'

const EXAMPLES = [
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Ethanol", smiles: "CCO" },
  { name: "Nitrobenzene", smiles: "c1ccc(cc1)N(=O)=O" },
  { name: "Pyrene (PAH)", smiles: "c1cc2ccc3cccc4ccc(c1)c2c34" },
  { name: "Glycerol", smiles: "OCC(O)CO" },
  { name: "Carbon Tetrachloride", smiles: "ClC(Cl)(Cl)Cl" },
]

export default function SMILESInput({ onPredict, loading }) {
  const [smiles, setSmiles] = useState('')
  const [name, setName] = useState('')
  const [showExamples, setShowExamples] = useState(false)

  const handleSubmit = () => { if (!smiles.trim()) return; onPredict(smiles.trim(), name.trim()) }
  const handleExample = (ex) => { setSmiles(ex.smiles); setName(ex.name); setShowExamples(false) }

  return (
    <div className="w-full max-w-3xl mx-auto animate-fade-in">
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 bg-accent/10 border border-accent/20 text-accent text-xs font-mono px-3 py-1.5 rounded-full mb-4">
          <FlaskConical size={12} /> ML-POWERED · RDKIT · TOX21
        </div>
        <h1 className="text-4xl font-mono font-bold text-white mb-3 tracking-tight">
          Tox<span className="text-accent">Predict</span>
        </h1>
        <p className="text-muted text-sm max-w-md mx-auto">
          Predict drug toxicity from molecular structure using machine learning
        </p>
      </div>

      <div className="bg-card border border-border rounded-2xl p-6 glow-accent">
        <label className="block text-xs font-mono text-muted mb-2 uppercase tracking-widest">SMILES String</label>
        <textarea
          value={smiles} onChange={e => setSmiles(e.target.value)}
          placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O" rows={2}
          className="w-full bg-surface border border-border rounded-xl px-4 py-3 text-white font-mono text-sm placeholder:text-muted/40 focus:outline-none focus:border-accent/60 resize-none transition-all"
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSubmit())}
        />
        <label className="block text-xs font-mono text-muted mt-4 mb-2 uppercase tracking-widest">
          Compound Name <span className="text-muted/50">(optional)</span>
        </label>
        <input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Aspirin"
          className="w-full bg-surface border border-border rounded-xl px-4 py-3 text-white text-sm placeholder:text-muted/40 focus:outline-none focus:border-accent/60 transition-all" />

        <div className="flex items-center gap-3 mt-5">
          <button onClick={handleSubmit} disabled={!smiles.trim() || loading}
            className="flex-1 flex items-center justify-center gap-2 bg-accent text-bg font-mono font-bold py-3 rounded-xl hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed transition-all text-sm">
            {loading ? <><div className="w-4 h-4 border-2 border-bg/40 border-t-bg rounded-full animate-spin" />ANALYZING...</> : <><Zap size={16} />PREDICT TOXICITY</>}
          </button>
          <button onClick={() => setShowExamples(v => !v)}
            className="flex items-center gap-2 border border-border text-muted hover:text-white hover:border-accent/40 font-mono text-xs px-4 py-3 rounded-xl transition-all">
            <Search size={14} />EXAMPLES{showExamples ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
        </div>

        {showExamples && (
          <div className="mt-4 grid grid-cols-2 gap-2 animate-fade-in">
            {EXAMPLES.map(ex => (
              <button key={ex.name} onClick={() => handleExample(ex)}
                className="text-left bg-surface border border-border hover:border-accent/40 rounded-xl px-4 py-3 transition-all group">
                <div className="text-white text-sm font-medium group-hover:text-accent transition-colors">{ex.name}</div>
                <div className="text-muted text-xs font-mono truncate mt-0.5">{ex.smiles}</div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
