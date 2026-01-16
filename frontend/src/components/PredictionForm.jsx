import { useState } from 'react'
import SampleButton from './SampleButton'

const CITIES = ['St. John\'s', 'Paradise', 'Mount Pearl', 'Torbay']

const HEATING_OPTIONS = [
  'Electric', 'Forced air', 'Heat pump', 'Baseboard heaters',
  'Radiant', 'Gas', 'Oil', 'Wood', 'Other'
]

const PARKING_OPTIONS = [
  'Attached Garage', 'Detached Garage', 'Carport', 'Parking pad',
  'No parking', 'Shared driveway', 'Street parking', 'Other'
]

const FEATURES_OPTIONS = [
  'Garage', 'Deck', 'Patio', 'Pool', 'Fireplace', 'Basement',
  'Central air', 'Security system', 'Storage', 'Workshop', 'Other'
]

const FLOORING_OPTIONS = [
  'Hardwood', 'Carpet', 'Tile', 'Laminate', 'Vinyl', 'Cork', 'Concrete', 'Other'
]

const EXTERIOR_OPTIONS = [
  'Vinyl siding', 'Brick', 'Wood', 'Stucco', 'Aluminum',
  'Fiber cement', 'Stone', 'Metal', 'Other'
]

const PARKING_FEATURES_OPTIONS = [
  'Attached Garage', 'Detached Garage', 'Carport', 'Parking pad',
  'No parking', 'Shared driveway', 'Street parking', 'Other'
]

function InputField({ label, name, type = 'text', value, onChange, required, options, tooltip, allowNegative }) {
  return (
    <div className="mb-4">
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </label>
      {options ? (
        <select
          name={name}
          value={value}
          onChange={onChange}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          required={required}
        >
          <option value="">Select...</option>
          {options.map(opt => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      ) : (
        <input
          type={type}
          name={name}
          value={value}
          onChange={onChange}
          step={type === 'number' ? 'any' : undefined}
          min={type === 'number' && !allowNegative ? '0' : undefined}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          required={required}
          placeholder={tooltip}
        />
      )}
      {tooltip && !options && (
        <p className="mt-1 text-xs text-gray-500">{tooltip}</p>
      )}
    </div>
  )
}

function PredictionForm({ onPredict, loading }) {
  const [formData, setFormData] = useState({
    address_locality: "St. John's",
    latitude: 47.5615,
    longitude: -52.7126,
    property_baths: 3,
    property_sqft: 1850,
    square_footage: 1850,
    heating: 'Electric',
    features: 'Garage',
    parking: 'Attached Garage',
    flooring: 'Hardwood',
    exterior: 'Vinyl siding',
    parking_features: 'Attached Garage'
  })

  const [errors, setErrors] = useState({})

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: name.includes('latitude') || name.includes('longitude') || name.includes('sqft') || name.includes('baths')
        ? parseFloat(value) || value
        : value
    }))
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: null }))
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    const newErrors = {}

    if (!formData.address_locality) newErrors.address_locality = 'Required'
    if (!formData.latitude || isNaN(formData.latitude)) newErrors.latitude = 'Valid number required'
    if (!formData.longitude || isNaN(formData.longitude)) newErrors.longitude = 'Valid number required'
    if (!formData.property_baths || formData.property_baths < 0) newErrors.property_baths = 'Valid number required'
    if (!formData.property_sqft || formData.property_sqft <= 0) newErrors.property_sqft = 'Valid number required'
    if (!formData.square_footage || formData.square_footage <= 0) newErrors.square_footage = 'Valid number required'

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }

    onPredict(formData)
  }

  const handleFillSample = () => {
    setFormData({
      address_locality: "St. John's",
      latitude: 47.5615,
      longitude: -52.7126,
      property_baths: 3,
      property_sqft: 1850,
      square_footage: 1850,
      heating: 'Electric',
      features: 'Garage',
      parking: 'Attached Garage',
      flooring: 'Hardwood',
      exterior: 'Vinyl siding',
      parking_features: 'Attached Garage'
    })
    setErrors({})
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900">Property Details</h2>
        <SampleButton onClick={handleFillSample} />
      </div>

      <form onSubmit={handleSubmit}>
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3 pb-2 border-b">
            Location
          </h3>
          <InputField
            label="City"
            name="address_locality"
            value={formData.address_locality}
            onChange={handleChange}
            required
            options={CITIES}
          />
          <div className="grid grid-cols-2 gap-4">
            <InputField
              label="Latitude"
              name="latitude"
              type="number"
              step="0.0001"
              value={formData.latitude}
              onChange={handleChange}
              required
              tooltip="Get from Google Maps: right-click location → copy coordinates"
            />
            <InputField
              label="Longitude"
              name="longitude"
              type="number"
              step="0.0001"
              allowNegative
              value={formData.longitude}
              onChange={handleChange}
              required
              tooltip="Get from Google Maps: right-click location → copy coordinates"
            />
          </div>
        </div>

        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3 pb-2 border-b">
            Size
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <InputField
              label="Property Sq Ft"
              name="property_sqft"
              type="number"
              value={formData.property_sqft}
              onChange={handleChange}
              required
              tooltip="Total property area in square feet"
            />
            <InputField
              label="Square Footage"
              name="square_footage"
              type="number"
              value={formData.square_footage}
              onChange={handleChange}
              required
              tooltip="Living area square footage"
            />
          </div>
          <InputField
            label="Bathrooms"
            name="property_baths"
            type="number"
            value={formData.property_baths}
            onChange={handleChange}
            required
            tooltip="Number of bathrooms"
          />
        </div>

        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3 pb-2 border-b">
            Features
          </h3>
          <InputField
            label="Heating"
            name="heating"
            value={formData.heating}
            onChange={handleChange}
            required
            options={HEATING_OPTIONS}
          />
          <InputField
            label="Features"
            name="features"
            value={formData.features}
            onChange={handleChange}
            required
            options={FEATURES_OPTIONS}
          />
          <InputField
            label="Parking"
            name="parking"
            value={formData.parking}
            onChange={handleChange}
            required
            options={PARKING_OPTIONS}
          />
          <InputField
            label="Flooring"
            name="flooring"
            value={formData.flooring}
            onChange={handleChange}
            required
            options={FLOORING_OPTIONS}
          />
          <InputField
            label="Exterior"
            name="exterior"
            value={formData.exterior}
            onChange={handleChange}
            required
            options={EXTERIOR_OPTIONS}
          />
          <InputField
            label="Parking Features"
            name="parking_features"
            value={formData.parking_features}
            onChange={handleChange}
            required
            options={PARKING_FEATURES_OPTIONS}
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-primary-600 text-white py-3 px-4 rounded-lg font-semibold hover:bg-primary-700 focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Predicting...
            </span>
          ) : (
            'Get Price Estimate'
          )}
        </button>
      </form>
    </div>
  )
}

export default PredictionForm
