import axios from 'axios'
const api = axios.create({ baseURL: '/api', timeout: 15000 })
export const predictToxicity = async (smiles, compoundName = '') => {
  const { data } = await api.post('/predict', { smiles, compound_name: compoundName || undefined })
  return data
}
export const predictBatch = async (compounds) => {
  const { data } = await api.post('/predict/batch', { compounds })
  return data
}
export const checkHealth = async () => {
  const { data } = await api.get('/health')
  return data
}
