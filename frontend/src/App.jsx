import { useState } from 'react'
import Header from './components/Header'
import PredictionForm from './components/PredictionForm'
import ResultCard from './components/ResultCard'

function App() {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handlePredict = async (formData) => {
    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const { predict } = await import('./services/api')
      const result = await predict(formData)
      setPrediction(result)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          <PredictionForm
            onPredict={handlePredict}
            loading={loading}
          />
          <ResultCard
            prediction={prediction}
            loading={loading}
            error={error}
          />
        </div>
      </main>
      <footer className="bg-white border-t mt-12 py-6">
        <div className="max-w-6xl mx-auto px-4 text-center text-gray-500 text-sm">
          <p>NL House Price Predictor - St. John's Metropolitan Area</p>
          <p className="mt-1">Powered by LightGBM Machine Learning</p>
        </div>
      </footer>
    </div>
  )
}

export default App
