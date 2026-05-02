import { useState, useEffect } from 'react'

const API = '/api'

export default function App() {
  const [scenarios, setScenarios] = useState([])
  const [algorithms, setAlgorithms] = useState([])
  const [selectedScenario, setSelectedScenario] = useState('')
  const [models, setModels] = useState([])
  const [progress, setProgress] = useState(null)
  const [isPolling, setIsPolling] = useState(false)
  const [loadError, setLoadError] = useState(null)
  const [config, setConfig] = useState({
    algorithm: 'maskable_ppo',
    n_envs: 4,
    total_timesteps: 100000,
    learning_rate: 0.0003,
    n_steps: 2048,
    batch_size: 64,
    n_epochs: 10,
    norm_obs: true,
    norm_reward: true,
    seed: 42,
    model_name: 'trained_model',
  })

  const loadInitialData = () => {
    setLoadError(null)
    Promise.all([
      fetch(`${API}/scenarios`).then(r => r.json()),
      fetch(`${API}/algorithms`).then(r => r.json()),
      fetch(`${API}/training/progress`).then(r => r.json()),
    ]).then(([scenariosData, algorithmsData, progressData]) => {
      setScenarios(scenariosData)
      setAlgorithms(algorithmsData)
      if (scenariosData.length > 0) {
        const first = scenariosData[0]
        setSelectedScenario(first.id)
        setConfig(c => ({ ...c, model_name: slugify(first.name) + '_model' }))
      }
      setProgress(progressData)
      if (progressData.status === 'running') setIsPolling(true)
    }).catch(() => setLoadError('Cannot reach the API. Is the backend running on port 8001?'))
  }

  // Initial data load
  useEffect(() => { loadInitialData() }, [])

  // Reload models when scenario changes
  useEffect(() => {
    if (!selectedScenario) return
    fetch(`${API}/scenarios/${selectedScenario}/models`).then(r => r.json()).then(setModels)
  }, [selectedScenario])

  // Polling loop — reschedules itself while isPolling is true
  useEffect(() => {
    if (!isPolling) return
    const id = setTimeout(() => {
      fetch(`${API}/training/progress`)
        .then(r => r.json())
        .then(prog => {
          setProgress(prog)
          if (prog.status !== 'running') {
            setIsPolling(false)
            if (prog.status === 'done' && selectedScenario) {
              fetch(`${API}/scenarios/${selectedScenario}/models`).then(r => r.json()).then(setModels)
            }
          }
        })
    }, 2000)
    return () => clearTimeout(id)
  }, [isPolling, progress, selectedScenario])

  const handleScenarioChange = (id) => {
    setSelectedScenario(id)
    const sc = scenarios.find(s => s.id === id)
    if (sc) setConfig(c => ({ ...c, model_name: slugify(sc.name) + '_model' }))
  }

  const startTraining = async () => {
    if (!selectedScenario) return
    const res = await fetch(`${API}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...config, scenario_id: selectedScenario }),
    })
    if (!res.ok) {
      const err = await res.json()
      alert(err.detail || 'Failed to start training')
      return
    }
    setProgress({ status: 'running', pct: 0, timesteps: 0, total_timesteps: config.total_timesteps, mean_reward: 0 })
    setIsPolling(true)
  }

  const deleteModel = async (modelId) => {
    await fetch(`${API}/scenarios/${selectedScenario}/models/${modelId}`, { method: 'DELETE' })
    fetch(`${API}/scenarios/${selectedScenario}/models`).then(r => r.json()).then(setModels)
  }

  const set = (key) => (e) => {
    const val = e.target.type === 'checkbox' ? e.target.checked : e.target.value
    setConfig(c => ({ ...c, [key]: val }))
  }

  const setNum = (key) => (e) => setConfig(c => ({ ...c, [key]: Number(e.target.value) }))

  const isTraining = progress?.status === 'running'
  const pct = progress?.pct ?? 0

  return (
    <div className="app">
      <header>
        <h1>ACES Platform</h1>
        <p>ACE Simulation Environment — Military Logistics Decision Support</p>
      </header>

      <main>
        {loadError && (
          <div className="error-banner" style={{ marginBottom: 16 }}>
            {loadError} <button className="btn-danger" style={{ marginLeft: 12 }} onClick={loadInitialData}>Retry</button>
          </div>
        )}

        <section className="card">
          <h2>Active Scenario</h2>
          <select value={selectedScenario} onChange={e => handleScenarioChange(e.target.value)}>
            {scenarios.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
        </section>

        <section className="card">
          <h2>Mission Planner — Train Agent</h2>

          <div className="grid-2">
            <label>
              Algorithm
              <select value={config.algorithm} onChange={set('algorithm')}>
                {algorithms.map(a => <option key={a.key} value={a.key}>{a.label}</option>)}
              </select>
            </label>
            <label>
              Parallel Environments
              <input type="number" min={1} max={16} value={config.n_envs} onChange={setNum('n_envs')} />
            </label>
          </div>

          <div className="grid-3">
            <label>
              Total Timesteps
              <input type="number" min={10000} max={10000000} step={10000}
                value={config.total_timesteps} onChange={setNum('total_timesteps')} />
            </label>
            <label>
              Learning Rate
              <input type="number" min={0.00001} max={0.01} step={0.00001}
                value={config.learning_rate} onChange={setNum('learning_rate')} />
            </label>
            <label>
              Random Seed (0 = random)
              <input type="number" min={0} max={99999} value={config.seed} onChange={setNum('seed')} />
            </label>
          </div>

          <div className="grid-3">
            <label>
              N Steps (PPO)
              <input type="number" min={128} max={8192} step={128}
                value={config.n_steps} onChange={setNum('n_steps')} />
            </label>
            <label>
              Batch Size
              <input type="number" min={32} max={1024} step={32}
                value={config.batch_size} onChange={setNum('batch_size')} />
            </label>
            <label>
              N Epochs (PPO)
              <input type="number" min={1} max={20}
                value={config.n_epochs} onChange={setNum('n_epochs')} />
            </label>
          </div>

          <div className="checkboxes">
            <label>
              <input type="checkbox" checked={config.norm_obs} onChange={set('norm_obs')} />
              Normalize Observations
            </label>
            <label>
              <input type="checkbox" checked={config.norm_reward} onChange={set('norm_reward')} />
              Normalize Rewards
            </label>
          </div>

          <label>
            Model Save Name
            <input type="text" value={config.model_name} onChange={set('model_name')} />
          </label>

          <button className="btn-primary" onClick={startTraining}
            disabled={isTraining || !selectedScenario || config.algorithm === 'heuristic_greedy'}>
            {isTraining ? 'Training…' : 'Start Training'}
          </button>
          {config.algorithm === 'heuristic_greedy' && (
            <p className="hint">Heuristic agents don't require training.</p>
          )}

          {progress && progress.status !== 'idle' && (
            <div className="progress-section">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${pct}%` }} />
              </div>
              <p className="progress-label">
                {(progress.timesteps ?? 0).toLocaleString()} / {(progress.total_timesteps ?? 0).toLocaleString()} steps
                &nbsp;({(pct).toFixed(1)}%)
              </p>
              <div className="metrics">
                <div className="metric">
                  <span className="metric-label">Mean Reward</span>
                  <span className="metric-value">{(progress.mean_reward ?? 0).toFixed(2)}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Status</span>
                  <span className={`metric-value status-${progress.status}`}>
                    {progress.status === 'running' ? 'Running'
                      : progress.status === 'done' ? '✓ Complete'
                      : `✗ ${progress.error}`}
                  </span>
                </div>
              </div>
              {progress.status === 'done' && (
                <div className="success-banner">
                  Training complete! Final mean reward: {(progress.mean_reward ?? 0).toFixed(2)}
                </div>
              )}
              {progress.status === 'error' && (
                <div className="error-banner">Training failed: {progress.error}</div>
              )}
            </div>
          )}
        </section>

        <section className="card">
          <h2>Saved Models</h2>
          {models.length === 0 ? (
            <p className="empty">No trained models yet for this scenario.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Algorithm</th>
                  <th>Steps</th>
                  <th>Mean Reward</th>
                  <th>Created</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {models.map(m => (
                  <tr key={m.id}>
                    <td>{m.name}</td>
                    <td>{m.algorithm}</td>
                    <td>{(m.training_steps ?? 0).toLocaleString()}</td>
                    <td>{parseFloat(m.mean_reward).toFixed(2)}</td>
                    <td>{(m.created_at ?? '').slice(0, 10)}</td>
                    <td>
                      <button className="btn-danger" onClick={() => deleteModel(m.id)}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </main>
    </div>
  )
}

function slugify(str) {
  return str.toLowerCase().replace(/\s+/g, '_').replace(/[^a-z0-9_]/g, '')
}
