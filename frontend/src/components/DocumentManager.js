import React, { useState, useEffect } from 'react';
import { 
  FileText, 
  Calendar, 
  Hash, 
  Trash2, 
  RefreshCw, 
  Download, 
  AlertTriangle,
  CheckCircle,
  Loader,
  Brain,
  Zap
} from 'lucide-react';
import { listDocuments, deleteDocuments, getStats } from '../api';
import toast from 'react-hot-toast';

const DocumentManager = ({ onDocumentDeleted }) => {
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
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
    } catch (error) {
      console.error('Failed to load data:', error);
      toast.error('Failed to load documents and statistics');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  const toggleDocumentSelection = (documentId) => {
    setSelectedDocuments(prev => 
      prev.includes(documentId)
        ? prev.filter(id => id !== documentId)
        : [...prev, documentId]
    );
  };

  const toggleSelectAll = () => {
    if (selectedDocuments.length === documents.length) {
      setSelectedDocuments([]);
    } else {
      setSelectedDocuments(documents.map(doc => doc.document_id));
    }
  };

  const handleDeleteSelected = async () => {
    if (selectedDocuments.length === 0) {
      toast.error('Please select documents to delete');
      return;
    }

    const confirmed = window.confirm(
      `Are you sure you want to delete ${selectedDocuments.length} document(s)? This action cannot be undone.`
    );

    if (!confirmed) return;

    setDeleting(true);
    try {
      const result = await deleteDocuments(selectedDocuments);
      
      if (result.success) {
        toast.success(result.message);
        setSelectedDocuments([]);
        await loadData();
        onDocumentDeleted?.(result);
      } else {
        toast.error(result.error || 'Failed to delete documents');
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to delete documents';
      toast.error(errorMessage);
    } finally {
      setDeleting(false);
    }
  };

  const formatTimestamp = (timestamp) => {
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

  const formatFileSize = (bytes) => {
    if (!bytes || bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileTypeIcon = (fileType) => {
    switch (fileType.toLowerCase()) {
      case 'pdf':
        return '📄';
      case 'docx':
      case 'doc':
        return '📝';
      case 'txt':
        return '📄';
      case 'md':
        return '📋';
      default:
        return '📄';
    }
  };

  const getIndexingModeBadge = (mode) => {
    const isAuto = mode === 'auto';
    return (
      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
        isAuto 
          ? 'bg-blue-100 text-blue-800' 
          : 'bg-purple-100 text-purple-800'
      }`}>
        <Brain className="h-3 w-3 mr-1" />
        {isAuto ? 'Auto' : 'Manual'}
      </span>
    );
  };

  const getEmbeddingModelBadge = (model) => {
    const isJina = model.includes('jina');
    return (
      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
        isJina 
          ? 'bg-green-100 text-green-800' 
          : 'bg-orange-100 text-orange-800'
      }`}>
        <Zap className="h-3 w-3 mr-1" />
        {model.split('/').pop()}
      </span>
    );
  };

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-12">
          <Loader className="h-8 w-8 text-primary-600 animate-spin" />
          <span className="ml-3 text-gray-600">Loading documents...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Statistics */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="card">
            <div className="flex items-center">
              <div className="p-2 bg-primary-100 rounded-lg">
                <Hash className="h-6 w-6 text-primary-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Chunks</p>
                <p className="text-2xl font-bold text-gray-900">{stats.total_chunks}</p>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="flex items-center">
              <div className="p-2 bg-green-100 rounded-lg">
                <FileText className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Documents</p>
                <p className="text-2xl font-bold text-gray-900">{stats.unique_documents}</p>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Brain className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Embedded</p>
                <p className="text-2xl font-bold text-gray-900">
                  {documents.reduce((sum, doc) => sum + (doc.embedded_chunks || 0), 0)}
                </p>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="flex items-center">
              <div className="p-2 bg-yellow-100 rounded-lg">
                <CheckCircle className="h-6 w-6 text-yellow-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Raw Content</p>
                <p className="text-2xl font-bold text-gray-900">
                  {documents.reduce((sum, doc) => sum + (doc.raw_chunks || 0), 0)}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Document List */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-medium text-gray-900">
            Documents ({documents.length})
          </h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="btn-secondary flex items-center space-x-2"
            >
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {documents.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No documents</h3>
            <p className="mt-1 text-sm text-gray-500">
              Upload some documents to get started.
            </p>
          </div>
        ) : (
          <>
            {/* Bulk Actions */}
            {selectedDocuments.length > 0 && (
              <div className="mb-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <AlertTriangle className="h-5 w-5 text-yellow-600" />
                    <span className="text-sm font-medium text-yellow-800">
                      {selectedDocuments.length} document(s) selected
                    </span>
                  </div>
                  <button
                    onClick={handleDeleteSelected}
                    disabled={deleting}
                    className="btn-danger flex items-center space-x-2"
                  >
                    {deleting ? (
                      <Loader className="h-4 w-4 animate-spin" />
                    ) : (
                      <Trash2 className="h-4 w-4" />
                    )}
                    <span>Delete Selected</span>
                  </button>
                </div>
              </div>
            )}

            {/* Documents Table */}
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left">
                      <input
                        type="checkbox"
                        checked={selectedDocuments.length === documents.length && documents.length > 0}
                        onChange={toggleSelectAll}
                        className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                      />
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Document
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Chunks
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Indexing
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Size
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Uploaded
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {documents.map((doc) => (
                    <tr key={doc.document_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <input
                          type="checkbox"
                          checked={selectedDocuments.includes(doc.document_id)}
                          onChange={() => toggleDocumentSelection(doc.document_id)}
                          className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                        />
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <span className="text-lg mr-2">
                            {getFileTypeIcon(doc.file_type)}
                          </span>
                          <div>
                            <div className="text-sm font-medium text-gray-900">
                              {doc.filename}
                            </div>
                            <div className="text-sm text-gray-500">
                              ID: {doc.document_id.slice(0, 8)}...
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                          {doc.file_type.toUpperCase()}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">
                          <div className="font-medium">{doc.chunks} total</div>
                          <div className="text-xs text-gray-500">
                            {doc.embedded_chunks || 0} embedded, {doc.raw_chunks || 0} raw
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="space-y-1">
                          {getIndexingModeBadge(doc.indexing_mode || 'auto')}
                          {getEmbeddingModelBadge(doc.embedding_model || 'unknown')}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatFileSize(doc.file_size)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatTimestamp(doc.upload_timestamp)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <button
                          onClick={() => toggleDocumentSelection(doc.document_id)}
                          className="text-red-600 hover:text-red-900"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default DocumentManager;