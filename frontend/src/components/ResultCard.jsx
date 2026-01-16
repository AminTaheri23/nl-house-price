function ResultCard({ prediction, loading, error }) {
  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-sm border p-6 h-fit">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Prediction Result</h2>
        <div className="flex flex-col items-center justify-center py-12">
          <svg className="animate-spin h-12 w-12 text-primary-600 mb-4" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          <p className="text-gray-500">Analyzing property data...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-white rounded-xl shadow-sm border p-6 h-fit">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Prediction Result</h2>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-red-700">
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <span className="font-medium">Prediction Failed</span>
          </div>
          <p className="mt-2 text-sm text-red-600">{error}</p>
        </div>
      </div>
    )
  }

  if (!prediction) {
    return (
      <div className="bg-white rounded-xl shadow-sm border p-6 h-fit">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Prediction Result</h2>
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <svg className="w-16 h-16 text-gray-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
          </svg>
          <p className="text-gray-500 mb-1">Enter property details</p>
          <p className="text-sm text-gray-400">Click "Get Price Estimate" to see the prediction</p>
        </div>
      </div>
    )
  }

  const qualityColors = {
    high: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-700', badge: 'bg-green-500' },
    medium: { bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-700', badge: 'bg-yellow-500' },
    low: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', badge: 'bg-red-500' }
  }

  const colors = qualityColors[prediction.quality] || qualityColors.medium

  return (
    <div className="bg-white rounded-xl shadow-sm border p-6 h-fit">
      <h2 className="text-xl font-semibold text-gray-900 mb-6">Prediction Result</h2>

      <div className={`rounded-xl p-6 ${colors.bg} border ${colors.border}`}>
        <div className="flex items-center justify-between mb-4">
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colors.badge} text-white`}>
            {prediction.quality_label}
          </span>
          <span className="text-sm text-gray-500">95% Confidence</span>
        </div>

        <div className="text-center mb-6">
          <p className="text-sm text-gray-600 mb-1">Estimated Price</p>
          <p className="text-4xl font-bold text-gray-900">
            ${prediction.predicted_price.toLocaleString()}
          </p>
        </div>

        <div className="border-t border-gray-200 pt-4">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-gray-500">Lower bound</span>
            <span className="font-medium text-gray-900">
              ${prediction.confidence_interval.lower.toLocaleString()}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
            <div
              className={`h-2 rounded-full ${colors.badge.replace('bg-', 'bg-')}`}
              style={{ width: '100%' }}
            />
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">Upper bound</span>
            <span className="font-medium text-gray-900">
              ${prediction.confidence_interval.upper.toLocaleString()}
            </span>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-gray-200">
          <p className="text-xs text-gray-500 text-center">
            Price range: ±${prediction.confidence_interval.margin.toLocaleString()}
          </p>
        </div>
      </div>

      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-sm font-semibold text-gray-900 mb-3">Model Performance</h3>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-xs text-gray-500">R² Score</p>
            <p className="font-semibold text-gray-900">{prediction.model_performance.r2_score}</p>
          </div>
          <div>
            <p className="text-xs text-gray-500">RMSE</p>
            <p className="font-semibold text-gray-900">
              ${prediction.model_performance.rmse.toLocaleString()}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500">MAE</p>
            <p className="font-semibold text-gray-900">
              ${prediction.model_performance.mae.toLocaleString()}
            </p>
          </div>
        </div>
      </div>

      <p className="mt-4 text-xs text-gray-400 text-center">
        Based on LightGBM model trained on {prediction.model_performance.n_features || 12} features
      </p>
    </div>
  )
}

export default ResultCard
