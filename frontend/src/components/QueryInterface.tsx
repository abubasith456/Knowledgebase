import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, 
  Filter, 
  FileText, 
  Calendar,
  Hash,
  TrendingUp
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { QueryInterfaceProps, SearchResult } from '../types';
import Button from './ui/Button';
import Card from './ui/Card';
import Badge from './ui/Badge';
import { cn } from '../utils/cn';

const QueryInterface: React.FC<QueryInterfaceProps> = ({ onSearchComplete }) => {
  const { uiState, searchDocuments, searchResults, documents } = useApp();
  const [query, setQuery] = useState('');
  const [nResults, setNResults] = useState(5);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [showDocumentFilter, setShowDocumentFilter] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    const documentIds = selectedDocuments.length > 0 ? selectedDocuments : undefined;
    await searchDocuments(query, nResults, documentIds);
    onSearchComplete?.(searchResults || []);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const toggleDocumentSelection = (documentId: string) => {
    setSelectedDocuments(prev => 
      prev.includes(documentId)
        ? prev.filter(id => id !== documentId)
        : [...prev, documentId]
    );
  };

  const formatTimestamp = (timestamp: string) => {
    if (!timestamp || timestamp === 'unknown') return 'Unknown';
    try {
      return new Date(timestamp).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return timestamp;
    }
  };

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.8) return 'success';
    if (similarity >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <div className="space-y-6">
      {/* Search Interface */}
      <Card variant="glass" className="p-6">
        <div className="flex items-center space-x-3 mb-6">
          <div className="p-2 bg-gradient-to-r from-primary-600 to-accent-600 rounded-lg">
            <Search className="h-5 w-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-900">Search Knowledge Base</h3>
            <p className="text-sm text-slate-600">Query your documents with natural language</p>
          </div>
        </div>

        <div className="space-y-4">
          {/* Search Input */}
          <div className="space-y-2">
            <label className="form-label">Search Query</label>
            <div className="relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Enter your search query..."
                className="input pr-12"
                disabled={uiState.isSearching}
              />
              <Button
                onClick={handleSearch}
                disabled={!query.trim() || uiState.isSearching}
                loading={uiState.isSearching}
                className="absolute right-2 top-1/2 transform -translate-y-1/2"
                size="sm"
              >
                <Search className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Search Options */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="form-label">Number of Results</label>
              <select
                value={nResults}
                onChange={(e) => setNResults(parseInt(e.target.value))}
                className="input"
                disabled={uiState.isSearching}
              >
                <option value={3}>3 results</option>
                <option value={5}>5 results</option>
                <option value={10}>10 results</option>
                <option value={15}>15 results</option>
                <option value={20}>20 results</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="form-label">Document Filter</label>
              <Button
                variant="secondary"
                onClick={() => setShowDocumentFilter(!showDocumentFilter)}
                className="w-full justify-between"
              >
                <span>
                  {selectedDocuments.length === 0 
                    ? 'All Documents' 
                    : `${selectedDocuments.length} selected`
                  }
                </span>
                <Filter className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Document Filter */}
          <AnimatePresence>
            {showDocumentFilter && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
                className="space-y-3"
              >
                <label className="form-label">Select Documents to Search</label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-40 overflow-y-auto">
                  {documents.map((doc) => (
                    <label
                      key={doc.document_id}
                      className="flex items-center space-x-2 p-2 rounded-lg hover:bg-slate-50 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={selectedDocuments.includes(doc.document_id)}
                        onChange={() => toggleDocumentSelection(doc.document_id)}
                        className="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
                      />
                      <span className="text-sm text-slate-700 truncate">{doc.filename}</span>
                    </label>
                  ))}
                </div>
                {documents.length === 0 && (
                  <p className="text-sm text-slate-500">No documents available</p>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </Card>

      {/* Search Results */}
      <AnimatePresence>
        {searchResults && searchResults.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="space-y-4"
          >
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-slate-900">
                Search Results ({searchResults.length})
              </h3>
              <Badge variant="primary">
                <TrendingUp className="h-3 w-3" />
                AI-Powered Search
              </Badge>
            </div>

            <div className="space-y-4">
              {searchResults.map((result, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card className="p-6 hover-lift">
                    <div className="space-y-4">
                      {/* Result Header */}
                      <div className="flex items-start justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="flex items-center justify-center w-8 h-8 bg-primary-100 rounded-lg">
                            <Hash className="h-4 w-4 text-primary-600" />
                          </div>
                          <div>
                            <h4 className="font-semibold text-slate-900">
                              Result #{result.index}
                            </h4>
                            <p className="text-sm text-slate-600">
                              From: {result.source} • Chunk: {result.chunk}
                            </p>
                          </div>
                        </div>
                        <Badge variant={getSimilarityColor(result.similarity) as any}>
                          {(result.similarity * 100).toFixed(1)}% match
                        </Badge>
                      </div>

                      {/* Result Content */}
                      <div className="bg-slate-50 rounded-lg p-4">
                        <p className="text-slate-800 leading-relaxed">
                          {result.content.length > 300 
                            ? `${result.content.substring(0, 300)}...`
                            : result.content
                          }
                        </p>
                      </div>

                      {/* Result Metadata */}
                      <div className="flex items-center justify-between text-sm text-slate-500">
                        <div className="flex items-center space-x-4">
                          <div className="flex items-center space-x-1">
                            <FileText className="h-3 w-3" />
                            <span>{result.metadata.file_type.toUpperCase()}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <Calendar className="h-3 w-3" />
                            <span>{formatTimestamp(result.metadata.upload_timestamp)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* No Results */}
      <AnimatePresence>
        {searchResults && searchResults.length === 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="text-center py-12">
              <Search className="mx-auto h-12 w-12 text-slate-400 mb-4" />
              <h3 className="text-lg font-medium text-slate-900 mb-2">No results found</h3>
              <p className="text-slate-600">
                Try adjusting your search query or document filter
              </p>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default QueryInterface;