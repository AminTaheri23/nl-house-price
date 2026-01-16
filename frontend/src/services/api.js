import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || '';

export const predict = async (data) => {
  const response = await axios.post(`${API_URL}/api/predict`, data, {
    headers: { 'Content-Type': 'application/json' }
  });
  return response.data;
};

export const getModelInfo = async () => {
  const response = await axios.get(`${API_URL}/api/info`);
  return response.data;
};

export const healthCheck = async () => {
  const response = await axios.get(`${API_URL}/api/health`);
  return response.data;
};
