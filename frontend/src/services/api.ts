import axios from 'axios';
import { 
  UploadResponse, 
  QueryResponse, 
  DocumentListResponse, 
  StatsResponse, 
  DeleteResponse,
  IndexingConfig 
} from '../types';

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

export const uploadDocument = async (file: File, indexingConfig: IndexingConfig = {}): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  
  // Add indexing configuration as query parameters
  const params = new URLSearchParams();
  params.append('indexing_mode', indexingConfig.mode || 'auto');
  params.append('embedding_model', indexingConfig.embedding_model || 'jinaai/jina-embeddings-v3');
  params.append('manual_chunk_size', (indexingConfig.manual_chunk_size || 1000).toString());
  params.append('manual_chunk_overlap', (indexingConfig.manual_chunk_overlap || 200).toString());
  params.append('auto_token_threshold', (indexingConfig.auto_token_threshold || 7000).toString());
  
  const response = await api.post<UploadResponse>(`/api/upload?${params.toString()}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const queryKnowledgeBase = async (
  query: string, 
  nResults: number = 5, 
  documentIds?: string[]
): Promise<QueryResponse> => {
  const response = await api.post<QueryResponse>('/api/query', {
    query,
    n_results: nResults,
    document_ids: documentIds,
  });
  return response.data;
};

export const getStats = async (): Promise<StatsResponse> => {
  const response = await api.get<StatsResponse>('/api/stats');
  return response.data;
};

export const listDocuments = async (): Promise<DocumentListResponse> => {
  const response = await api.get<DocumentListResponse>('/api/documents');
  return response.data;
};

export const deleteDocuments = async (documentIds: string[]): Promise<DeleteResponse> => {
  const response = await api.delete<DeleteResponse>('/api/documents', {
    data: { document_ids: documentIds },
  });
  return response.data;
};

export const healthCheck = async (): Promise<{ status: string; service: string }> => {
  const response = await api.get('/api/health');
  return response.data;
};

export default api;