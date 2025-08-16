import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  AppContextType, 
  UIState, 
  Document, 
  StatsResponse, 
  SearchResult, 
  IndexingConfig,
  UploadResponse,
  QueryResponse,
  DeleteResponse
} from '../types';
import { 
  uploadDocument as apiUploadDocument,
  queryKnowledgeBase,
  listDocuments,
  getStats,
  deleteDocuments as apiDeleteDocuments
} from '../api';
import toast from 'react-hot-toast';

// Initial state
const initialUIState: UIState = {
  isLoading: false,
  isUploading: false,
  isSearching: false,
  activeTab: 'upload',
  showAdvancedOptions: false,
  selectedDocuments: [],
  refreshTrigger: 0,
};

const initialIndexingConfig: IndexingConfig = {
  mode: 'auto',
  embedding_model: 'jinaai/jina-embeddings-v3',
  manual_chunk_size: 1000,
  manual_chunk_overlap: 200,
  auto_token_threshold: 7000,
};

// Reducer for complex state management
type AppAction = 
  | { type: 'SET_UI_STATE'; payload: Partial<UIState> }
  | { type: 'SET_DOCUMENTS'; payload: Document[] }
  | { type: 'SET_STATS'; payload: StatsResponse | null }
  | { type: 'SET_SEARCH_RESULTS'; payload: SearchResult[] | null }
  | { type: 'SET_INDEXING_CONFIG'; payload: Partial<IndexingConfig> }
  | { type: 'TOGGLE_DOCUMENT_SELECTION'; payload: string }
  | { type: 'SELECT_ALL_DOCUMENTS' }
  | { type: 'CLEAR_DOCUMENT_SELECTION' }
  | { type: 'CLEAR_SEARCH_RESULTS' }
  | { type: 'INCREMENT_REFRESH_TRIGGER' };

interface AppState {
  uiState: UIState;
  documents: Document[];
  stats: StatsResponse | null;
  searchResults: SearchResult[] | null;
  indexingConfig: IndexingConfig;
}

const initialState: AppState = {
  uiState: initialUIState,
  documents: [],
  stats: null,
  searchResults: null,
  indexingConfig: initialIndexingConfig,
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_UI_STATE':
      return {
        ...state,
        uiState: { ...state.uiState, ...action.payload },
      };
    
    case 'SET_DOCUMENTS':
      return {
        ...state,
        documents: action.payload,
      };
    
    case 'SET_STATS':
      return {
        ...state,
        stats: action.payload,
      };
    
    case 'SET_SEARCH_RESULTS':
      return {
        ...state,
        searchResults: action.payload,
      };
    
    case 'SET_INDEXING_CONFIG':
      return {
        ...state,
        indexingConfig: { ...state.indexingConfig, ...action.payload },
      };
    
    case 'TOGGLE_DOCUMENT_SELECTION':
      const selectedDocuments = state.uiState.selectedDocuments.includes(action.payload)
        ? state.uiState.selectedDocuments.filter(id => id !== action.payload)
        : [...state.uiState.selectedDocuments, action.payload];
      return {
        ...state,
        uiState: { ...state.uiState, selectedDocuments },
      };
    
    case 'SELECT_ALL_DOCUMENTS':
      return {
        ...state,
        uiState: {
          ...state.uiState,
          selectedDocuments: state.documents.map(doc => doc.document_id),
        },
      };
    
    case 'CLEAR_DOCUMENT_SELECTION':
      return {
        ...state,
        uiState: { ...state.uiState, selectedDocuments: [] },
      };
    
    case 'CLEAR_SEARCH_RESULTS':
      return {
        ...state,
        searchResults: null,
      };
    
    case 'INCREMENT_REFRESH_TRIGGER':
      return {
        ...state,
        uiState: {
          ...state.uiState,
          refreshTrigger: state.uiState.refreshTrigger + 1,
        },
      };
    
    default:
      return state;
  }
}

// Create context
const AppContext = createContext<AppContextType | undefined>(undefined);

// Provider component
export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Load initial data
  useEffect(() => {
    refreshData();
  }, []);

  // Actions
  const setUIState = (uiState: Partial<UIState>) => {
    dispatch({ type: 'SET_UI_STATE', payload: uiState });
  };

  const setDocuments = (documents: Document[]) => {
    dispatch({ type: 'SET_DOCUMENTS', payload: documents });
  };

  const setStats = (stats: StatsResponse | null) => {
    dispatch({ type: 'SET_STATS', payload: stats });
  };

  const setSearchResults = (results: SearchResult[] | null) => {
    dispatch({ type: 'SET_SEARCH_RESULTS', payload: results });
  };

  const setIndexingConfig = (config: Partial<IndexingConfig>) => {
    dispatch({ type: 'SET_INDEXING_CONFIG', payload: config });
  };

  const toggleDocumentSelection = (documentId: string) => {
    dispatch({ type: 'TOGGLE_DOCUMENT_SELECTION', payload: documentId });
  };

  const selectAllDocuments = () => {
    dispatch({ type: 'SELECT_ALL_DOCUMENTS' });
  };

  const clearDocumentSelection = () => {
    dispatch({ type: 'CLEAR_DOCUMENT_SELECTION' });
  };

  const clearSearchResults = () => {
    dispatch({ type: 'CLEAR_SEARCH_RESULTS' });
  };

  // API Actions
  const uploadDocument = async (file: File) => {
    try {
      setUIState({ isUploading: true });
      
      const result = await apiUploadDocument(file, state.indexingConfig);
      
      if (result.success) {
        toast.success('Document uploaded successfully!');
        await refreshData();
      } else {
        toast.error(result.error || 'Upload failed');
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'Upload failed';
      toast.error(errorMessage);
    } finally {
      setUIState({ isUploading: false });
    }
  };

  const searchDocuments = async (query: string, nResults: number = 5, documentIds?: string[]) => {
    try {
      setUIState({ isSearching: true });
      
      const result = await queryKnowledgeBase(query, nResults, documentIds || undefined);
      
      if (result.success) {
        const searchResults: SearchResult[] = result.results.documents[0].map((document: string, index: number) => {
          const metadata = result.results.metadatas[0][index];
          const distance = result.results.distances[0][index];
          const similarity = 1 - distance;
          
          return {
            index: index + 1,
            source: metadata.source,
            chunk: `${parseInt(metadata.chunk_index) + 1}/${metadata.total_chunks}`,
            similarity,
            content: document,
            metadata: {
              file_type: metadata.file_type,
              upload_timestamp: metadata.upload_timestamp,
            },
          };
        });
        
        setSearchResults(searchResults);
        toast.success(`Found ${result.count} results`);
      } else {
        toast.error(result.error || 'Search failed');
        setSearchResults(null);
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'Search failed';
      toast.error(errorMessage);
      setSearchResults(null);
    } finally {
      setUIState({ isSearching: false });
    }
  };

  const deleteDocuments = async (documentIds: string[]) => {
    try {
      setUIState({ isLoading: true });
      
      const result = await apiDeleteDocuments(documentIds);
      
      if (result.success) {
        toast.success(result.message);
        clearDocumentSelection();
        await refreshData();
      } else {
        toast.error(result.error || 'Failed to delete documents');
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to delete documents';
      toast.error(errorMessage);
    } finally {
      setUIState({ isLoading: false });
    }
  };

  const refreshData = async () => {
    try {
      setUIState({ isLoading: true });
      
      const [docsResponse, statsResponse] = await Promise.all([
        listDocuments(),
        getStats()
      ]);

      if (docsResponse.success) {
        setDocuments(docsResponse.documents);
      }

      if (statsResponse && !statsResponse.error) {
        setStats(statsResponse);
      }
      
      dispatch({ type: 'INCREMENT_REFRESH_TRIGGER' });
    } catch (error: any) {
      console.error('Failed to refresh data:', error);
      toast.error('Failed to load data');
    } finally {
      setUIState({ isLoading: false });
    }
  };

  const contextValue: AppContextType = {
    // State
    uiState: state.uiState,
    documents: state.documents,
    stats: state.stats,
    searchResults: state.searchResults,
    indexingConfig: state.indexingConfig,
    
    // Actions
    setUIState,
    setDocuments,
    setStats,
    setSearchResults,
    setIndexingConfig,
    
    // API Actions
    uploadDocument,
    searchDocuments,
    deleteDocuments,
    refreshData,
    
    // Utility Actions
    clearSearchResults,
    toggleDocumentSelection,
    selectAllDocuments,
    clearDocumentSelection,
  };

  return (
    <AppContext.Provider value={contextValue}>
      <AnimatePresence>
        {children}
      </AnimatePresence>
    </AppContext.Provider>
  );
}

// Hook to use the context
export function useApp() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}