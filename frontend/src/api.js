import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth headers if needed
api.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

export const uploadDocument = async (file, indexingConfig = {}) => {
  const formData = new FormData();
  formData.append('file', file);
  
  // Add indexing configuration as query parameters
  const params = new URLSearchParams();
  params.append('indexing_mode', indexingConfig.mode || 'auto');
  params.append('embedding_model', indexingConfig.embedding_model || 'jinaai/jina-embeddings-v3');
  params.append('manual_chunk_size', indexingConfig.manual_chunk_size || 1000);
  params.append('manual_chunk_overlap', indexingConfig.manual_chunk_overlap || 200);
  params.append('auto_token_threshold', indexingConfig.auto_token_threshold || 7000);
  
  const response = await api.post(`/api/upload?${params.toString()}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const queryKnowledgeBase = async (query, nResults = 5, documentIds = null) => {
  const response = await api.post('/api/query', {
    query,
    n_results: nResults,
    document_ids: documentIds,
  });
  return response.data;
};

export const getStats = async () => {
  const response = await api.get('/api/stats');
  return response.data;
};

export const listDocuments = async () => {
  const response = await api.get('/api/documents');
  return response.data;
};

export const deleteDocuments = async (documentIds) => {
  const response = await api.delete('/api/documents', {
    data: { document_ids: documentIds },
  });
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

export default api;