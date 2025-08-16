import React, { useState, useEffect } from 'react';
import { Search, Loader, FileText, Calendar, Hash, Trash2 } from 'lucide-react';
import { queryKnowledgeBase, listDocuments } from '../api';
import toast from 'react-hot-toast';

const QueryInterface = () => {
  const [query, setQuery] = useState('');
  const [nResults, setNResults] = useState(5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [showDocumentFilter, setShowDocumentFilter] = useState(false);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const response = await listDocuments();
      if (response.success) {
        setDocuments(response.documents);
      }
    } catch (error) {
      console.error('Failed to load documents:', error);
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      toast.error('Please enter a query');
      return;
    }

    setLoading(true);
    try {
      const documentIds = selectedDocuments.length > 0 ? selectedDocuments : null;
      const result = await queryKnowledgeBase(query, nResults, documentIds);
      
      if (result.success) {
        setResults(result);
        toast.success(`Found ${result.count} results`);
      } else {
        toast.error(result.error || 'Search failed');
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Search failed';
      toast.error(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  const toggleDocumentSelection = (documentId) => {
    setSelectedDocuments(prev => 
      prev.includes(documentId)
        ? prev.filter(id => id !== documentId)
        : [...prev, documentId]
    );
  };

  const clearResults = () => {
    setResults(null);
    setQuery('');
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp || timestamp === 'unknown') return 'Unknown';
    try {
      return new Date(timestamp).toLocaleDateString();
    } catch {
      return timestamp;
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes || bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      {/* Search Input */}
      <div className="card">
        <div className="space-y-4">
          <div>
            <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
              Search Query
            </label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter your question or search terms..."
              className="input-field resize-none"
              rows={3}
              disabled={loading}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label htmlFor="nResults" className="block text-sm font-medium text-gray-700 mb-2">
                Number of Results
              </label>
              <select
                id="nResults"
                value={nResults}
                onChange={(e) => setNResults(parseInt(e.target.value))}
                className="input-field"
                disabled={loading}
              >
                {[1, 3, 5, 10, 15, 20].map(num => (
                  <option key={num} value={num}>{num}</option>
                ))}
              </select>
            </div>

            <div className="flex items-end">
              <button
                onClick={() => setShowDocumentFilter(!showDocumentFilter)}
                className="btn-secondary w-full"
                disabled={loading}
              >
                {selectedDocuments.length > 0 
                  ? `${selectedDocuments.length} document(s) selected`
                  : 'Filter by documents'
                }
              </button>
            </div>

            <div className="flex items-end space-x-2">
              <button
                onClick={handleSearch}
                disabled={loading || !query.trim()}
                className="btn-primary flex-1 flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <Loader className="h-4 w-4 animate-spin" />
                ) : (
                  <Search className="h-4 w-4" />
                )}
                <span>Search</span>
              </button>
              {results && (
                <button
                  onClick={clearResults}
                  className="btn-secondary px-3"
                  title="Clear results"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Document Filter */}
      {showDocumentFilter && (
        <div className="card">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Filter by Documents</h3>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {documents.length === 0 ? (
              <p className="text-gray-500 text-sm">No documents available</p>
            ) : (
              documents.map((doc) => (
                <label key={doc.document_id} className="flex items-center space-x-3 p-2 rounded hover:bg-gray-50 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedDocuments.includes(doc.document_id)}
                    onChange={() => toggleDocumentSelection(doc.document_id)}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">{doc.filename}</p>
                    <div className="flex items-center space-x-4 text-xs text-gray-500">
                      <span className="flex items-center">
                        <FileText className="h-3 w-3 mr-1" />
                        {doc.file_type}
                      </span>
                      <span className="flex items-center">
                        <Hash className="h-3 w-3 mr-1" />
                        {doc.chunks} chunks
                      </span>
                      <span className="flex items-center">
                        <Calendar className="h-3 w-3 mr-1" />
                        {formatTimestamp(doc.upload_timestamp)}
                      </span>
                    </div>
                  </div>
                </label>
              ))
            )}
          </div>
          {selectedDocuments.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <button
                onClick={() => setSelectedDocuments([])}
                className="text-sm text-red-600 hover:text-red-700 flex items-center"
              >
                <Trash2 className="h-4 w-4 mr-1" />
                Clear selection
              </button>
            </div>
          )}
        </div>
      )}

      {/* Search Results */}
      {results && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">
              Search Results ({results.count})
            </h3>
            {results.filtered_by_documents && (
              <span className="text-sm text-gray-500">
                Filtered by {results.filtered_by_documents.length} document(s)
              </span>
            )}
          </div>

          {results.count === 0 ? (
            <div className="text-center py-8">
              <Search className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No results found</h3>
              <p className="mt-1 text-sm text-gray-500">
                Try adjusting your search terms or filters.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {results.results.documents[0].map((document, index) => {
                const metadata = results.results.metadatas[0][index];
                const distance = results.results.distances[0][index];
                const similarity = 1 - distance;

                return (
                  <div key={index} className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800">
                          #{index + 1}
                        </span>
                        <span className="text-sm font-medium text-gray-900">
                          {metadata.source}
                        </span>
                        <span className="text-sm text-gray-500">
                          (Chunk {parseInt(metadata.chunk_index) + 1}/{metadata.total_chunks})
                        </span>
                      </div>
                      <span className="text-sm font-medium text-gray-600">
                        {(similarity * 100).toFixed(1)}% match
                      </span>
                    </div>
                    
                    <div className="text-sm text-gray-700 leading-relaxed">
                      {document.length > 300 
                        ? `${document.substring(0, 300)}...`
                        : document
                      }
                    </div>
                    
                    <div className="mt-2 flex items-center space-x-4 text-xs text-gray-500">
                      <span>Type: {metadata.file_type}</span>
                      <span>Uploaded: {formatTimestamp(metadata.upload_timestamp)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QueryInterface;