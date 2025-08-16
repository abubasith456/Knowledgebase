// API Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface UploadResponse {
  success: boolean;
  file_name: string;
  token_count: number;
  chunks_created: number;
  chunks_deleted: number;
  needs_chunking: boolean;
  action: string;
  message: string;
  indexing_mode: string;
  embedding_model: string;
  error?: string;
}

export interface QueryResponse {
  success: boolean;
  query: string;
  results: {
    documents: string[][];
    metadatas: any[][];
    distances: number[][];
  };
  count: number;
  filtered_by_documents?: string[];
  error?: string;
}

export interface Document {
  filename: string;
  chunks: number;
  file_type: string;
  upload_timestamp: string;
  document_hash: string;
  document_id: string;
  file_size: number;
  indexing_mode: string;
  embedding_model: string;
  embedded_chunks: number;
  raw_chunks: number;
}

export interface DocumentListResponse {
  success: boolean;
  documents: Document[];
  total_count: number;
  error?: string;
}

export interface StatsResponse {
  total_chunks: number;
  unique_documents: number;
  documents: string[];
  collection_name: string;
  error?: string;
}

export interface DeleteResponse {
  success: boolean;
  deleted_documents: number;
  deleted_chunks: number;
  document_names: string[];
  message: string;
  error?: string;
}

// Indexing Configuration Types
export interface IndexingConfig {
  mode: 'auto' | 'manual';
  embedding_model: 'jinaai/jina-embeddings-v3' | 'qwen3-0.6B';
  manual_chunk_size: number;
  manual_chunk_overlap: number;
  auto_token_threshold: number;
}

// Search Result Types
export interface SearchResult {
  index: number;
  source: string;
  chunk: string;
  similarity: number;
  content: string;
  metadata: {
    file_type: string;
    upload_timestamp: string;
  };
}

// UI State Types
export interface UIState {
  isLoading: boolean;
  isUploading: boolean;
  isSearching: boolean;
  activeTab: string;
  showAdvancedOptions: boolean;
  selectedDocuments: string[];
  refreshTrigger: number;
  sidebarOpen: boolean;
}

// App Context Types
export interface AppContextType {
  // State
  uiState: UIState;
  documents: Document[];
  stats: StatsResponse | null;
  searchResults: SearchResult[] | null;
  indexingConfig: IndexingConfig;
  
  // Actions
  setUIState: (state: Partial<UIState>) => void;
  setDocuments: (documents: Document[]) => void;
  setStats: (stats: StatsResponse | null) => void;
  setSearchResults: (results: SearchResult[] | null) => void;
  setIndexingConfig: (config: Partial<IndexingConfig>) => void;
  
  // API Actions
  uploadDocument: (file: File) => Promise<void>;
  searchDocuments: (query: string, nResults?: number, documentIds?: string[]) => Promise<void>;
  deleteDocuments: (documentIds: string[]) => Promise<void>;
  refreshData: () => Promise<void>;
  
  // Utility Actions
  clearSearchResults: () => void;
  toggleDocumentSelection: (documentId: string) => void;
  selectAllDocuments: () => void;
  clearDocumentSelection: () => void;
  toggleSidebar: () => void;
}

// Component Props Types
export interface FileUploadProps {
  onUploadSuccess?: (result: UploadResponse) => void;
  onUploadError?: (error: any) => void;
}

export interface QueryInterfaceProps {
  onSearchComplete?: (results: SearchResult[]) => void;
}

export interface DocumentManagerProps {
  onDocumentDeleted?: (result: DeleteResponse) => void;
}

export interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate: (id: string) => void;
  activeId?: string;
}

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

export interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'success';
  size?: 'sm' | 'md' | 'lg';
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  loading?: boolean;
  className?: string;
}

export interface CardProps {
  variant?: 'default' | 'glass' | 'gradient';
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
}

export interface BadgeProps {
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'accent';
  children: React.ReactNode;
  className?: string;
}

// Utility Types
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface ToastMessage {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  duration?: number;
}

// Theme Types
export interface Theme {
  mode: 'light' | 'dark';
  primary: string;
  accent: string;
}

// Settings Types
export interface AppSettings {
  theme: Theme;
  autoRefresh: boolean;
  refreshInterval: number;
  maxUploadSize: number;
  defaultIndexingConfig: IndexingConfig;
}

// Navigation Types
export interface NavItem {
  id: string;
  name: string;
  icon: React.ComponentType<any>;
  description: string;
  component?: React.ReactNode;
}