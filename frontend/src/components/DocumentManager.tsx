import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FileText, 
  Trash2, 
  RefreshCw, 
  AlertTriangle,
  CheckCircle,
  Loader2,
  Brain,
  Zap,
  Database,
  Calendar,
  Hash
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { DocumentManagerProps } from '../types';
import Button from './ui/Button';
import Card from './ui/Card';
import Badge from './ui/Badge';
import { cn } from '../utils/cn';

const DocumentManager: React.FC<DocumentManagerProps> = ({ onDocumentDeleted }) => {
  const { 
    uiState, 
    documents, 
    stats, 
    toggleDocumentSelection, 
    selectAllDocuments, 
    clearDocumentSelection,
    deleteDocuments,
    refreshData
  } = useApp();

  const handleDeleteSelected = async () => {
    if (uiState.selectedDocuments.length === 0) return;

    const confirmed = window.confirm(
      `Are you sure you want to delete ${uiState.selectedDocuments.length} document(s)? This action cannot be undone.`
    );

    if (!confirmed) return;

    await deleteDocuments(uiState.selectedDocuments);
    onDocumentDeleted?.({ success: true } as any);
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

  const formatFileSize = (bytes: number) => {
    if (!bytes || bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileTypeIcon = (fileType: string) => {
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

  const getIndexingModeBadge = (mode: string) => {
    const isAuto = mode === 'auto';
    return (
      <Badge variant={isAuto ? 'primary' : 'accent'}>
        <Brain className="h-3 w-3" />
        {isAuto ? 'Auto' : 'Manual'}
      </Badge>
    );
  };

  const getEmbeddingModelBadge = (model: string) => {
    const isJina = model.includes('jina');
    return (
      <Badge variant={isJina ? 'success' : 'warning'}>
        <Zap className="h-3 w-3" />
        {model.split('/').pop()}
      </Badge>
    );
  };

  if (uiState.isLoading) {
    return (
      <Card className="flex items-center justify-center py-12">
        <div className="text-center">
          <Loader2 className="mx-auto h-8 w-8 text-primary-600 animate-spin mb-4" />
          <p className="text-slate-600">Loading documents...</p>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Statistics */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="flex items-center">
              <div className="p-2 bg-primary-100 rounded-lg">
                <Hash className="h-6 w-6 text-primary-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-slate-600">Total Chunks</p>
                <p className="text-2xl font-bold text-slate-900">{stats.total_chunks}</p>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center">
              <div className="p-2 bg-success-100 rounded-lg">
                <FileText className="h-6 w-6 text-success-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-slate-600">Documents</p>
                <p className="text-2xl font-bold text-slate-900">{stats.unique_documents}</p>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Brain className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-slate-600">Embedded</p>
                <p className="text-2xl font-bold text-slate-900">
                  {documents.reduce((sum, doc) => sum + (doc.embedded_chunks || 0), 0)}
                </p>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center">
              <div className="p-2 bg-warning-100 rounded-lg">
                <CheckCircle className="h-6 w-6 text-warning-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-slate-600">Raw Content</p>
                <p className="text-2xl font-bold text-slate-900">
                  {documents.reduce((sum, doc) => sum + (doc.raw_chunks || 0), 0)}
                </p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Document List */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-r from-primary-600 to-accent-600 rounded-lg">
              <Database className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">
                Documents ({documents.length})
              </h3>
              <p className="text-sm text-slate-600">Manage your uploaded documents</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="secondary"
              onClick={refreshData}
              disabled={uiState.isLoading}
              loading={uiState.isLoading}
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
          </div>
        </div>

        {documents.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="mx-auto h-12 w-12 text-slate-400 mb-4" />
            <h3 className="text-lg font-medium text-slate-900 mb-2">No documents</h3>
            <p className="text-slate-600">
              Upload some documents to get started.
            </p>
          </div>
        ) : (
          <>
            {/* Bulk Actions */}
            <AnimatePresence>
              {uiState.selectedDocuments.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="mb-4 p-4 bg-warning-50 border border-warning-200 rounded-xl"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <AlertTriangle className="h-5 w-5 text-warning-600" />
                      <span className="text-sm font-medium text-warning-800">
                        {uiState.selectedDocuments.length} document(s) selected
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={clearDocumentSelection}
                      >
                        Clear
                      </Button>
                      <Button
                        variant="danger"
                        size="sm"
                        onClick={handleDeleteSelected}
                        loading={uiState.isLoading}
                      >
                        <Trash2 className="h-4 w-4" />
                        Delete Selected
                      </Button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Documents Table */}
            <div className="overflow-x-auto">
              <table className="table-modern">
                <thead>
                  <tr>
                    <th className="w-12">
                      <input
                        type="checkbox"
                        checked={uiState.selectedDocuments.length === documents.length && documents.length > 0}
                        onChange={uiState.selectedDocuments.length === documents.length ? clearDocumentSelection : selectAllDocuments}
                        className="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
                      />
                    </th>
                    <th>Document</th>
                    <th>Type</th>
                    <th>Chunks</th>
                    <th>Indexing</th>
                    <th>Size</th>
                    <th>Uploaded</th>
                    <th className="w-12">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((doc, index) => (
                    <motion.tr
                      key={doc.document_id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="hover:bg-slate-50/50 transition-colors duration-200"
                    >
                      <td>
                        <input
                          type="checkbox"
                          checked={uiState.selectedDocuments.includes(doc.document_id)}
                          onChange={() => toggleDocumentSelection(doc.document_id)}
                          className="rounded border-slate-300 text-primary-600 focus:ring-primary-500"
                        />
                      </td>
                      <td>
                        <div className="flex items-center">
                          <span className="text-lg mr-3">
                            {getFileTypeIcon(doc.file_type)}
                          </span>
                          <div>
                            <div className="font-medium text-slate-900">
                              {doc.filename}
                            </div>
                            <div className="text-sm text-slate-500">
                              ID: {doc.document_id.slice(0, 8)}...
                            </div>
                          </div>
                        </div>
                      </td>
                      <td>
                        <Badge variant="secondary">
                          {doc.file_type.toUpperCase()}
                        </Badge>
                      </td>
                      <td>
                        <div className="text-sm">
                          <div className="font-medium">{doc.chunks} total</div>
                          <div className="text-slate-500">
                            {doc.embedded_chunks || 0} embedded, {doc.raw_chunks || 0} raw
                          </div>
                        </div>
                      </td>
                      <td>
                        <div className="space-y-1">
                          {getIndexingModeBadge(doc.indexing_mode || 'auto')}
                          {getEmbeddingModelBadge(doc.embedding_model || 'unknown')}
                        </div>
                      </td>
                      <td className="text-sm text-slate-900">
                        {formatFileSize(doc.file_size)}
                      </td>
                      <td className="text-sm text-slate-500">
                        {formatTimestamp(doc.upload_timestamp)}
                      </td>
                      <td>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleDocumentSelection(doc.document_id)}
                          className="text-error-600 hover:text-error-700"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Card>
    </div>
  );
};

export default DocumentManager;