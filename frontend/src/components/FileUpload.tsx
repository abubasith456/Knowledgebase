import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  FileText, 
  AlertCircle, 
  CheckCircle, 
  Loader2, 
  Settings, 
  Brain,
  Zap,
  Sparkles
} from 'lucide-react';
import { useApp } from '../context/AppContext';
import { FileUploadProps } from '../types';
import Button from './ui/Button';
import Card from './ui/Card';
import Badge from './ui/Badge';
import { cn } from '../utils/cn';

const FileUpload: React.FC<FileUploadProps> = ({ onUploadSuccess, onUploadError }) => {
  const { uiState, setUIState, indexingConfig, setIndexingConfig, uploadDocument } = useApp();
  const [uploadStatus, setUploadStatus] = useState<any>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploadStatus(null);

    try {
      await uploadDocument(file);
      setUploadStatus({
        type: 'success',
        message: 'Document uploaded successfully!',
      });
      onUploadSuccess?.({ success: true } as any);
    } catch (error: any) {
      setUploadStatus({
        type: 'error',
        message: error.message || 'Upload failed',
      });
      onUploadError?.(error);
    }
  }, [uploadDocument, onUploadSuccess, onUploadError]);

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
    disabled: uiState.isUploading,
  });

  const handleConfigChange = (key: string, value: any) => {
    setIndexingConfig({ [key]: value });
  };

  return (
    <div className="space-y-6">
      {/* Indexing Configuration */}
      <Card variant="glass" className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-r from-primary-600 to-accent-600 rounded-lg">
              <Settings className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Indexing Configuration</h3>
              <p className="text-sm text-slate-600">Configure how your documents are processed</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setUIState({ showAdvancedOptions: !uiState.showAdvancedOptions })}
          >
            {uiState.showAdvancedOptions ? 'Hide' : 'Show'} Advanced
          </Button>
        </div>

        <AnimatePresence>
          {uiState.showAdvancedOptions && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Indexing Mode */}
                <div className="space-y-2">
                  <label className="form-label">Indexing Mode</label>
                  <select
                    value={indexingConfig.mode}
                    onChange={(e) => handleConfigChange('mode', e.target.value)}
                    className="input"
                  >
                    <option value="auto">Auto (Smart Token Detection)</option>
                    <option value="manual">Manual (Character-based Chunking)</option>
                  </select>
                  <p className="form-help">
                    {indexingConfig.mode === 'auto' 
                      ? 'Automatically detects if content needs chunking based on token count'
                      : 'Manually chunks content by character size with overlap'
                    }
                  </p>
                </div>

                {/* Embedding Model */}
                <div className="space-y-2">
                  <label className="form-label">Embedding Model</label>
                  <select
                    value={indexingConfig.embedding_model}
                    onChange={(e) => handleConfigChange('embedding_model', e.target.value)}
                    className="input"
                  >
                    <option value="jinaai/jina-embeddings-v3">Jina Embeddings v3 (Default)</option>
                    <option value="qwen3-0.6B">Qwen3 0.6B</option>
                  </select>
                  <p className="form-help">AI model used for generating text embeddings</p>
                </div>

                {/* Auto Mode Settings */}
                {indexingConfig.mode === 'auto' && (
                  <div className="space-y-2">
                    <label className="form-label">Token Threshold</label>
                    <input
                      type="number"
                      value={indexingConfig.auto_token_threshold}
                      onChange={(e) => handleConfigChange('auto_token_threshold', parseInt(e.target.value))}
                      className="input"
                      min="1000"
                      max="10000"
                      step="500"
                    />
                    <p className="form-help">Content below this threshold skips embedding (stored as raw text)</p>
                  </div>
                )}

                {/* Manual Mode Settings */}
                {indexingConfig.mode === 'manual' && (
                  <>
                    <div className="space-y-2">
                      <label className="form-label">Chunk Size (characters)</label>
                      <input
                        type="number"
                        value={indexingConfig.manual_chunk_size}
                        onChange={(e) => handleConfigChange('manual_chunk_size', parseInt(e.target.value))}
                        className="input"
                        min="100"
                        max="5000"
                        step="100"
                      />
                      <p className="form-help">Maximum characters per chunk</p>
                    </div>
                    <div className="space-y-2">
                      <label className="form-label">Chunk Overlap (characters)</label>
                      <input
                        type="number"
                        value={indexingConfig.manual_chunk_overlap}
                        onChange={(e) => handleConfigChange('manual_chunk_overlap', parseInt(e.target.value))}
                        className="input"
                        min="0"
                        max="1000"
                        step="50"
                      />
                      <p className="form-help">Overlap between consecutive chunks</p>
                    </div>
                  </>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Current Configuration Summary */}
        <motion.div 
          className="mt-6 p-4 bg-gradient-to-r from-primary-50 to-accent-50 rounded-xl border border-primary-100"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center space-x-3">
            <Brain className="h-5 w-5 text-primary-600" />
            <div className="flex-1">
              <p className="text-sm font-medium text-slate-900">Current Configuration</p>
              <p className="text-sm text-slate-600">
                {indexingConfig.mode === 'auto' ? 'Auto' : 'Manual'} mode with {indexingConfig.embedding_model.split('/').pop()}
                {indexingConfig.mode === 'auto' && ` (threshold: ${indexingConfig.auto_token_threshold} tokens)`}
                {indexingConfig.mode === 'manual' && ` (chunk: ${indexingConfig.manual_chunk_size} chars, overlap: ${indexingConfig.manual_chunk_overlap} chars)`}
              </p>
            </div>
            <Badge variant="primary">
              <Zap className="h-3 w-3" />
              {indexingConfig.mode === 'auto' ? 'Smart' : 'Manual'}
            </Badge>
          </div>
        </motion.div>
      </Card>

      {/* File Upload Area */}
      <Card variant="gradient" className="p-8">
        <div
          {...getRootProps()}
          className={cn(
            "border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300",
            isDragActive && !isDragReject
              ? "border-primary-500 bg-primary-50/50"
              : isDragReject
              ? "border-error-500 bg-error-50/50"
              : "border-slate-300 hover:border-primary-400 hover:bg-white/20",
            uiState.isUploading && "opacity-50 cursor-not-allowed"
          )}
        >
          <input {...getInputProps()} />
          
          <AnimatePresence mode="wait">
            {uiState.isUploading ? (
              <motion.div
                key="uploading"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="space-y-4"
              >
                <div className="relative">
                  <Loader2 className="mx-auto h-16 w-16 text-primary-600 animate-spin" />
                  <motion.div
                    className="absolute inset-0 rounded-full border-4 border-primary-200"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-slate-900 mb-2">Processing Document</h3>
                  <p className="text-slate-600">Please wait while we analyze and store your document</p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="upload-ready"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="space-y-6"
              >
                <div className="relative">
                  <div className="mx-auto w-20 h-20 bg-gradient-to-r from-primary-600 to-accent-600 rounded-2xl flex items-center justify-center shadow-glow">
                    <Upload className="h-10 w-10 text-white" />
                  </div>
                  <motion.div
                    className="absolute -top-2 -right-2 w-6 h-6 bg-success-500 rounded-full flex items-center justify-center"
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  >
                    <Sparkles className="h-3 w-3 text-white" />
                  </motion.div>
                </div>
                
                <div className="space-y-2">
                  <h3 className="text-2xl font-bold text-slate-900">
                    {isDragActive && !isDragReject
                      ? 'Drop your document here'
                      : isDragReject
                      ? 'File type not supported'
                      : 'Upload Your Document'
                    }
                  </h3>
                  <p className="text-slate-600">
                    {isDragActive && !isDragReject
                      ? 'Release to upload'
                      : isDragReject
                      ? 'Please select a supported file type'
                      : 'Drag & drop a document here, or click to select'
                    }
                  </p>
                  <p className="text-sm text-slate-500">
                    Supports PDF, DOCX, DOC, TXT, and MD files
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </Card>

      {/* Upload Status */}
      <AnimatePresence>
        {uploadStatus && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Card className={cn(
              uploadStatus.type === 'success' ? 'border-success-200 bg-success-50/50' : 'border-error-200 bg-error-50/50'
            )}>
              <div className="flex items-start space-x-4">
                {uploadStatus.type === 'success' ? (
                  <CheckCircle className="h-6 w-6 text-success-600 mt-1" />
                ) : (
                  <AlertCircle className="h-6 w-6 text-error-600 mt-1" />
                )}
                <div className="flex-1">
                  <h3 className={cn(
                    "font-semibold",
                    uploadStatus.type === 'success' ? 'text-success-800' : 'text-error-800'
                  )}>
                    {uploadStatus.type === 'success' ? 'Upload Successful' : 'Upload Failed'}
                  </h3>
                  <p className={cn(
                    "text-sm mt-1",
                    uploadStatus.type === 'success' ? 'text-success-700' : 'text-error-700'
                  )}>
                    {uploadStatus.message}
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default FileUpload;