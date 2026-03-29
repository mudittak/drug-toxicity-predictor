import { AlertTriangle, CheckCircle, XCircle, Info, Activity } from 'lucide-react'

const RISK_CONFIG = {
  HIGH:     { color: 'text-danger',     bg: 'bg-danger/10',     border: 'border-danger/30',     glow: 'glow-danger', icon: XCircle,      label: 'HIGH RISK' },
  MODERATE: { color: 'text-warn',       bg: 'bg-warn/10',       border: 'border-warn/30',       glow: '',            icon: AlertTriangle, label: 'MODERATE RISK' },
  LOW:      { color: 'text-yellow-400', bg: 'bg-yellow-400/10', border: 'border-yellow-400/30', glow: '',            icon: AlertTriangle, label: 'LOW RISK' },
  MINIMAL:  { color: 'text-safe',       bg: 'bg-safe/10',       border: 'border-safe/30',       glow: 'glow-safe',   icon: CheckCircle,   label: 'MINIMAL RISK' },
}

function PropRow({ label, value, highlight }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
      <span className="text-muted text-xs">{label}</span>
      <span className={'text-xs font-mono font-bold ' + (highlight ? 'text-warn' : 'text-white')}>
        {value}
      </span>
    </div>
  )
}

function MoleculeViewer({ smiles }) {
  const imageUrl = '/api/molecule-image/' + encodeURIComponent(smiles)

  return (
    <div className="bg-card border border-border rounded-2xl p-5">
      <div className="flex items-center gap-2 mb-4">
        <span className="text-accent text-xs">⬡</span>
        <h3 className="text-xs font-mono text-accent uppercase tracking-widest">
          2D Molecular Structure
        </h3>
      </div>
      <div className="flex items-center justify-center bg-white rounded-xl p-4">
        <img
          src={imageUrl}
          alt="Molecular Structure"
          className="max-w-full h-auto rounded"
          style={{ maxHeight: '220px' }}
          onError={e => {
            e.target.style.display = 'none'
            e.target.nextSibling.style.display = 'flex'
          }}
        />
        <div
          className="hidden items-center justify-center text-muted text-xs font-mono p-8"
          style={{ display: 'none' }}
        >
          Structure unavailable
        </div>
      </div>
      <p className="text-muted text-xs font-mono text-center mt-3 truncate">{smiles}</p>
    </div>
  )
}

export default function ResultCard({ result }) {
  if (!result) return null

  if (!result.valid) {
    return (
      <div className="w-full max-w-3xl mx-auto mt-6 animate-slide-up">
        <div className="bg-danger/10 border border-danger/30 rounded-2xl p-6 text-center">
          <XCircle className="text-danger mx-auto mb-3" size={32} />
          <p className="text-danger font-mono font-bold">INVALID SMILES</p>
          <p className="text-muted text-sm mt-1">{result.error}</p>
        </div>
      </div>
    )
  }

  const config = RISK_CONFIG[result.risk_level] || RISK_CONFIG.MINIMAL
  const Icon   = config.icon
  const props  = result.molecular_properties || {}

  return (
    <div className="w-full max-w-3xl mx-auto mt-8 animate-slide-up space-y-4">

      {/* Main Result Banner */}
      <div className={config.bg + ' border ' + config.border + ' ' + config.glow + ' rounded-2xl p-6'}>
        <div className="flex items-center justify-between mb-4">
          <div>
            {result.compound_name && (
              <p className="text-muted text-xs font-mono uppercase tracking-widest mb-1">
                {result.compound_name}
              </p>
            )}
            <div className="flex items-center gap-3">
              <Icon className={config.color} size={28} />
              <div>
                <p className={'text-2xl font-mono font-bold ' + config.color}>{result.prediction}</p>
                <p className={'text-xs font-mono opacity-70 ' + config.color}>{config.label}</p>
              </div>
            </div>
          </div>
          <div className="text-right">
            <p className={'text-4xl font-mono font-bold ' + config.color}>{result.toxic_probability}%</p>
            <p className="text-muted text-xs font-mono">TOXIC PROBABILITY</p>
          </div>
        </div>

        {/* Probability Bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs font-mono text-muted">
            <span>SAFE</span><span>TOXIC</span>
          </div>
          <div className="flex gap-1 h-3 rounded-full overflow-hidden bg-surface">
            <div className="bg-safe  transition-all duration-1000" style={{ width: result.safe_probability + '%' }} />
            <div className="bg-danger transition-all duration-1000" style={{ width: result.toxic_probability + '%' }} />
          </div>
          <div className="flex justify-between text-xs font-mono">
            <span className="text-safe">{result.safe_probability}%</span>
            <span className="text-danger">{result.toxic_probability}%</span>
          </div>
        </div>
      </div>

      {/* Molecule 2D Structure — full width */}
      <MoleculeViewer smiles={result.smiles} />

      {/* Properties + Factors */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* Molecular Properties */}
        <div className="bg-card border border-border rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-4">
            <Activity size={14} className="text-accent" />
            <h3 className="text-xs font-mono text-accent uppercase tracking-widest">Molecular Properties</h3>
          </div>
          <PropRow label="Molecular Weight"  value={props.molecular_weight + ' g/mol'} highlight={props.molecular_weight > 500} />
          <PropRow label="LogP"              value={props.logP}                         highlight={props.logP > 5} />
          <PropRow label="H-Bond Donors"     value={props.h_bond_donors}                highlight={props.h_bond_donors > 5} />
          <PropRow label="H-Bond Acceptors"  value={props.h_bond_acceptors}             highlight={props.h_bond_acceptors > 10} />
          <PropRow label="TPSA"              value={props.tpsa + ' A2'}                 highlight={props.tpsa < 20} />
          <PropRow label="Aromatic Rings"    value={props.aromatic_rings}               highlight={props.aromatic_rings >= 3} />
          <PropRow label="Heavy Atoms"       value={props.heavy_atom_count} />
          <div className={'mt-4 text-center py-2 rounded-lg text-xs font-mono font-bold ' + (props.lipinski_violations <= 1 ? 'bg-safe/10 text-safe' : 'bg-warn/10 text-warn')}>
            {props.drug_likeness} · {props.lipinski_violations} Lipinski violation{props.lipinski_violations !== 1 ? 's' : ''}
          </div>
        </div>

        {/* Key Factors */}
        <div className="bg-card border border-border rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-4">
            <Info size={14} className="text-accent" />
            <h3 className="text-xs font-mono text-accent uppercase tracking-widest">Key Factors</h3>
          </div>
          <div className="space-y-3">
            {(result.toxicity_factors || []).map((factor, i) => (
              <div key={i} className="flex items-start gap-3 bg-surface rounded-xl px-3 py-2.5">
                <div className={'w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ' + config.color.replace('text-', 'bg-')} />
                <p className="text-white text-xs leading-relaxed">{factor}</p>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-border">
            <p className="text-muted text-xs font-mono">MODEL</p>
            <p className="text-white text-xs mt-1">{result.model}</p>
          </div>
        </div>
      </div>

      {/* SMILES display */}
      <div className="bg-card border border-border rounded-xl px-5 py-3 flex items-center gap-3">
        <span className="text-muted text-xs font-mono uppercase tracking-widest flex-shrink-0">SMILES</span>
        <span className="text-accent text-xs font-mono truncate">{result.smiles}</span>
      </div>
    </div>
  )
}