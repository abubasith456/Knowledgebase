import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle, Loader, Settings, Brain } from 'lucide-react';
import { uploadDocument } from '../api';
import toast from 'react-hot-toast';

const FileUpload = ({ onUploadSuccess, onUploadError }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  
  // Indexing configuration state
  const [indexingConfig, setIndexingConfig] = useState({
    mode: 'auto',
    embedding_model: 'jinaai/jina-embeddings-v3',
    manual_chunk_size: 1000,
    manual_chunk_overlap: 200,
    auto_token_threshold: 7000,
  });

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploading(true);
    setUploadStatus(null);

    try {
      const result = await uploadDocument(file, indexingConfig);
      
      if (result.success) {
        setUploadStatus({
          type: 'success',
          message: result.message,
          details: {
            fileName: result.file_name,
            tokenCount: result.token_count,
            chunksCreated: result.chunks_created,
            chunksDeleted: result.chunks_deleted,
            action: result.action,
            indexingMode: result.indexing_mode,
            embeddingModel: result.embedding_model,
          },
        });
        toast.success('Document uploaded successfully!');
        onUploadSuccess?.(result);
      } else {
        setUploadStatus({
          type: 'error',
          message: result.error || 'Upload failed',
        });
        toast.error(result.error || 'Upload failed');
        onUploadError?.(result);
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Upload failed';
      setUploadStatus({
        type: 'error',
        message: errorMessage,
      });
      toast.error(errorMessage);
      onUploadError?.({ error: errorMessage });
    } finally {
      setUploading(false);
    }
  }, [onUploadSuccess, onUploadError, indexingConfig]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
    },
    multiple: false,
    disabled: uploading,
  });

  const handleConfigChange = (key, value) => {
    setIndexingConfig(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      {/* Advanced Options */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 flex items-center">
            <Settings className="h-5 w-5 mr-2 text-primary-600" />
            Indexing Configuration
          </h3>
          <button
            onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
            className="text-sm text-primary-600 hover:text-primary-700"
          >
            {showAdvancedOptions ? 'Hide' : 'Show'} Advanced Options
          </button>
        </div>

        {showAdvancedOptions && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Indexing Mode */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Indexing Mode
              </label>
              <select
                value={indexingConfig.mode}
                onChange={(e) => handleConfigChange('mode', e.target.value)}
                className="input-field"
              >
                <option value="auto">Auto (Smart Token Detection)</option>
                <option value="manual">Manual (Character-based Chunking)</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {indexingConfig.mode === 'auto' 
                  ? 'Automatically detects if content needs chunking based on token count'
                  : 'Manually chunks content by character size with overlap'
                }
              </p>
            </div>

            {/* Embedding Model */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Embedding Model
              </label>
              <select
                value={indexingConfig.embedding_model}
                onChange={(e) => handleConfigChange('embedding_model', e.target.value)}
                className="input-field"
              >
                <option value="jinaai/jina-embeddings-v3">Jina Embeddings v3 (Default)</option>
                <option value="qwen3-0.6B">Qwen3 0.6B</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                AI model used for generating text embeddings
              </p>
            </div>

            {/* Auto Mode Settings */}
            {indexingConfig.mode === 'auto' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Token Threshold
                </label>
                <input
                  type="number"
                  value={indexingConfig.auto_token_threshold}
                  onChange={(e) => handleConfigChange('auto_token_threshold', parseInt(e.target.value))}
                  className="input-field"
                  min="1000"
                  max="10000"
                  step="500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Content below this threshold skips embedding (stored as raw text)
                </p>
              </div>
            )}

            {/* Manual Mode Settings */}
            {indexingConfig.mode === 'manual' && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Chunk Size (characters)
                  </label>
                  <input
                    type="number"
                    value={indexingConfig.manual_chunk_size}
                    onChange={(e) => handleConfigChange('manual_chunk_size', parseInt(e.target.value))}
                    className="input-field"
                    min="100"
                    max="5000"
                    step="100"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Maximum characters per chunk
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Chunk Overlap (characters)
                  </label>
                  <input
                    type="number"
                    value={indexingConfig.manual_chunk_overlap}
                    onChange={(e) => handleConfigChange('manual_chunk_overlap', parseInt(e.target.value))}
                    className="input-field"
                    min="0"
                    max="1000"
                    step="50"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Overlap between consecutive chunks
                  </p>
                </div>
              </>
            )}
          </div>
        )}

        {/* Current Configuration Summary */}
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <div className="flex items-center text-sm text-gray-600">
            <Brain className="h-4 w-4 mr-2" />
            <span className="font-medium">Current Configuration:</span>
            <span className="ml-2">
              {indexingConfig.mode === 'auto' ? 'Auto' : 'Manual'} mode with {indexingConfig.embedding_model.split('/').pop()}
              {indexingConfig.mode === 'auto' && ` (threshold: ${indexingConfig.auto_token_threshold} tokens)`}
              {indexingConfig.mode === 'manual' && ` (chunk: ${indexingConfig.manual_chunk_size} chars, overlap: ${indexingConfig.manual_chunk_overlap} chars)`}
            </span>
          </div>
        </div>
      </div>

      {/* File Upload Area */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200
          ${isDragActive && !isDragReject
            ? 'border-primary-500 bg-primary-50'
            : isDragReject
            ? 'border-red-500 bg-red-50'
            : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          }
          ${uploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        {uploading ? (
          <div className="space-y-2">
            <Loader className="mx-auto h-12 w-12 text-primary-600 animate-spin" />
            <p className="text-lg font-medium text-gray-700">Processing document...</p>
            <p className="text-sm text-gray-500">Please wait while we analyze and store your document</p>
          </div>
        ) : (
          <div className="space-y-4">
            <Upload className="mx-auto h-12 w-12 text-gray-400" />
            <div>
              <p className="text-lg font-medium text-gray-700">
                {isDragActive && !isDragReject
                  ? 'Drop your document here'
                  : isDragReject
                  ? 'File type not supported'
                  : 'Drag & drop a document here, or click to select'
                }
              </p>
              <p className="text-sm text-gray-500 mt-1">
                Supports PDF, DOCX, DOC, TXT, and MD files
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Upload Status */}
      {uploadStatus && (
        <div className={`card ${uploadStatus.type === 'success' ? 'border-green-200' : 'border-red-200'}`}>
          <div className="flex items-start space-x-3">
            {uploadStatus.type === 'success' ? (
              <CheckCircle className="h-6 w-6 text-green-600 mt-0.5" />
            ) : (
              <AlertCircle className="h-6 w-6 text-red-600 mt-0.5" />
            )}
            <div className="flex-1">
              <h3 className={`font-medium ${uploadStatus.type === 'success' ? 'text-green-800' : 'text-red-800'}`}>
                {uploadStatus.type === 'success' ? 'Upload Successful' : 'Upload Failed'}
              </h3>
              <p className={`text-sm mt-1 ${uploadStatus.type === 'success' ? 'text-green-700' : 'text-red-700'}`}>
                {uploadStatus.message}
              </p>
              
              {uploadStatus.details && (
                <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium text-gray-600">File:</span>
                    <span className="ml-2 text-gray-800">{uploadStatus.details.fileName}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Action:</span>
                    <span className="ml-2 text-gray-800 capitalize">{uploadStatus.details.action}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Tokens:</span>
                    <span className="ml-2 text-gray-800">{uploadStatus.details.tokenCount.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Chunks:</span>
                    <span className="ml-2 text-gray-800">
                      {uploadStatus.details.chunksCreated} created
                      {uploadStatus.details.chunksDeleted > 0 && `, ${uploadStatus.details.chunksDeleted} deleted`}
                    </span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Mode:</span>
                    <span className="ml-2 text-gray-800 capitalize">{uploadStatus.details.indexingMode}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600">Model:</span>
                    <span className="ml-2 text-gray-800">{uploadStatus.details.embeddingModel.split('/').pop()}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;