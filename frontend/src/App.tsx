import { useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type TablePreview = Record<string, unknown>[]

type Recommendation = {
  part_id: string
  expected_qty: number
  recommended_qty: number
}

type Scenario = {
  scenario: string
  money_cost: number
  downtime_hours: number
  satisfaction_penalty: number
  combined_loss: number
}

type SummaryResponse = {
  events_evaluated: number
  baseline: Record<string, number>
  recommended: Record<string, number>
  delta_vs_baseline: Record<string, number>
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

function App() {
  const [status, setStatus] = useState<string>('')
  const [artifactId, setArtifactId] = useState<string>('')
  const [dealerCasesPreview, setDealerCasesPreview] = useState<TablePreview>([])
  const [eventsPreview, setEventsPreview] = useState<TablePreview>([])
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [summary, setSummary] = useState<SummaryResponse | null>(null)
  const [eventIdForSim, setEventIdForSim] = useState<string>('E000001')
  const [form, setForm] = useState({
    symptom_text: 'engine knocking and power loss under heavy load',
    task_type: 'material_handling',
    duration_days: 3,
    age_days: 1100,
    avg_load_factor: 0.92,
    environment_score: 0.66,
    region: 'north',
    safety_buffer: 0.35,
  })

  const scenarioMax = useMemo(() => {
    if (scenarios.length === 0) return 1
    return Math.max(...scenarios.map((s) => s.combined_loss), 1)
  }, [scenarios])

  const fetchJson = async <T,>(path: string, options?: RequestInit): Promise<T> => {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(errorText || `Request failed (${response.status})`)
    }
    return (await response.json()) as T
  }

  const handleGenerateData = async () => {
    try {
      setStatus('Generating demo data...')
      await fetchJson('/generate-demo-data', {
        method: 'POST',
        body: JSON.stringify({ n_events: 2000, seed: 42, demo_events: 50 }),
      })
      const dealer = await fetchJson<{ rows: TablePreview }>('/oracle/table/dealer_cases?limit=5')
      const events = await fetchJson<{ rows: TablePreview }>('/oracle/table/events?limit=5')
      setDealerCasesPreview(dealer.rows)
      setEventsPreview(events.rows)
      setStatus('Data generated and preview loaded.')
    } catch (error) {
      setStatus(`Generate data failed: ${String(error)}`)
    }
  }

  const handleTrainModel = async () => {
    try {
      setStatus('Training model...')
      const result = await fetchJson<{ artifact_id: string }>('/train-model', {
        method: 'POST',
        body: JSON.stringify({}),
      })
      setArtifactId(result.artifact_id)
      setStatus(`Model trained: ${result.artifact_id}`)
    } catch (error) {
      setStatus(`Training failed: ${String(error)}`)
    }
  }

  const handleRecommend = async (e: FormEvent) => {
    e.preventDefault()
    try {
      setStatus('Running recommendation...')
      const payload = {
        ...form,
        artifact_id: artifactId || undefined,
      }
      const result = await fetchJson<{ recommendations: Recommendation[] }>('/recommend-parts', {
        method: 'POST',
        body: JSON.stringify(payload),
      })
      setRecommendations(result.recommendations)
      setStatus('Recommendation complete.')
    } catch (error) {
      setStatus(`Recommend failed: ${String(error)}`)
    }
  }

  const handleSimulate = async () => {
    try {
      setStatus('Running simulation...')
      const payload =
        eventIdForSim.trim().length > 0
          ? { event_id: eventIdForSim, artifact_id: artifactId || undefined }
          : { ...form, artifact_id: artifactId || undefined }

      const result = await fetchJson<{ scenarios: Scenario[]; recommendations: Recommendation[] }>(
        '/simulate',
        {
          method: 'POST',
          body: JSON.stringify(payload),
        },
      )
      setScenarios(result.scenarios)
      setRecommendations(result.recommendations)
      setStatus('Simulation complete.')
    } catch (error) {
      setStatus(`Simulation failed: ${String(error)}`)
    }
  }

  const handleSummary = async () => {
    try {
      setStatus('Loading executive summary...')
      const result = await fetchJson<SummaryResponse>('/exec-summary')
      setSummary(result)
      setStatus('Executive summary loaded.')
    } catch (error) {
      setStatus(`Executive summary failed: ${String(error)}`)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Kubota Predictive Parts Demo</h1>
        <p className="subtitle">API: {API_BASE}</p>
        <p className="status">{status || 'Ready.'}</p>
      </header>

      <section className="panel">
        <h2>Generate Data</h2>
        <button onClick={handleGenerateData}>Generate Demo Data</button>
        <div className="preview-grid">
          <div>
            <h3>dealer_cases (sample)</h3>
            <pre>{JSON.stringify(dealerCasesPreview, null, 2)}</pre>
          </div>
          <div>
            <h3>events (sample)</h3>
            <pre>{JSON.stringify(eventsPreview, null, 2)}</pre>
          </div>
        </div>
      </section>

      <section className="panel">
        <h2>Train Model</h2>
        <button onClick={handleTrainModel}>Train Model</button>
        <p>Artifact ID: {artifactId || 'not trained yet'}</p>
      </section>

      <section className="panel">
        <h2>Case Intake</h2>
        <form onSubmit={handleRecommend} className="intake-form">
          <textarea
            value={form.symptom_text}
            onChange={(e) => setForm({ ...form, symptom_text: e.target.value })}
          />
          <div className="form-row">
            <input
              value={form.task_type}
              onChange={(e) => setForm({ ...form, task_type: e.target.value })}
              placeholder="task_type"
            />
            <input
              type="number"
              value={form.duration_days}
              onChange={(e) => setForm({ ...form, duration_days: Number(e.target.value) })}
              placeholder="duration_days"
            />
            <input
              type="number"
              value={form.age_days}
              onChange={(e) => setForm({ ...form, age_days: Number(e.target.value) })}
              placeholder="age_days"
            />
          </div>
          <div className="form-row">
            <input
              type="number"
              step="0.01"
              value={form.avg_load_factor}
              onChange={(e) => setForm({ ...form, avg_load_factor: Number(e.target.value) })}
              placeholder="avg_load_factor"
            />
            <input
              type="number"
              step="0.01"
              value={form.environment_score}
              onChange={(e) => setForm({ ...form, environment_score: Number(e.target.value) })}
              placeholder="environment_score"
            />
            <input
              value={form.region}
              onChange={(e) => setForm({ ...form, region: e.target.value })}
              placeholder="region"
            />
          </div>
          <button type="submit">Recommend Parts</button>
        </form>
      </section>

      <section className="panel">
        <h2>Simulation Results</h2>
        <div className="form-row">
          <input
            value={eventIdForSim}
            onChange={(e) => setEventIdForSim(e.target.value)}
            placeholder="event_id (optional)"
          />
          <button onClick={handleSimulate}>Run Simulation</button>
        </div>
        {scenarios.map((scenario) => (
          <div key={scenario.scenario} className="bar-row">
            <span>{scenario.scenario}</span>
            <div className="bar-wrap">
              <div
                className="bar"
                style={{ width: `${Math.max(4, (scenario.combined_loss / scenarioMax) * 100)}%` }}
              />
            </div>
            <span>{scenario.combined_loss.toFixed(1)}</span>
          </div>
        ))}
      </section>

      <section className="panel">
        <h2>Executive Summary</h2>
        <button onClick={handleSummary}>Load Executive Summary</button>
        <pre>{summary ? JSON.stringify(summary, null, 2) : 'No summary loaded.'}</pre>
      </section>

      <section className="panel">
        <h2>Current Recommendations</h2>
        <div className="table-like">
          <div className="table-head">
            <span>Part</span>
            <span>Expected Qty</span>
            <span>Recommended Qty</span>
          </div>
          {recommendations.map((r) => (
            <div className="table-row" key={r.part_id}>
              <span>{r.part_id}</span>
              <span>{r.expected_qty.toFixed(2)}</span>
              <span>{r.recommended_qty}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

export default App
