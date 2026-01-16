import { useState, useEffect } from 'react'
import { getModelInfo } from '../services/api'

function Header() {
  const [modelInfo, setModelInfo] = useState(null)
  const [showInfo, setShowInfo] = useState(false)

  useEffect(() => {
    getModelInfo()
      .then(setModelInfo)
      .catch(() => setModelInfo(null))
  }, [])

  return (
    <header className="bg-white border-b shadow-sm">
      <div className="max-w-6xl mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <svg className="w-10 h-10 text-primary-600" fill="currentColor" viewBox="0 0 24 24">
              <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/>
            </svg>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">NL House Price Predictor</h1>
              <p className="text-sm text-gray-500">St. John's Metropolitan Area</p>
            </div>
          </div>
          <button
            onClick={() => setShowInfo(!showInfo)}
            className="text-sm text-primary-600 hover:text-primary-700 font-medium"
          >
            {showInfo ? 'Hide Model Info' : 'Model Info'}
          </button>
        </div>

        {showInfo && modelInfo && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg border">
            <h3 className="font-semibold text-gray-900 mb-2">Model Performance</h3>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-gray-500">R² Score</p>
                <p className="font-semibold text-gray-900">{modelInfo.performance.r2_score}</p>
              </div>
              <div>
                <p className="text-gray-500">RMSE</p>
                <p className="font-semibold text-gray-900">${modelInfo.performance.rmse.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-gray-500">MAE</p>
                <p className="font-semibold text-gray-900">${modelInfo.performance.mae.toLocaleString()}</p>
              </div>
            </div>
            <p className="mt-3 text-xs text-gray-400">
              Based on {modelInfo.n_features} features including location, size, and property characteristics.
            </p>
          </div>
        )}
      </div>
    </header>
  )
}

export default Header
