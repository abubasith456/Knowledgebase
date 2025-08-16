import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle, Loader } from 'lucide-react';
import { uploadDocument } from '../api';
import toast from 'react-hot-toast';

const FileUpload = ({ onUploadSuccess, onUploadError }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploading(true);
    setUploadStatus(null);

    try {
      const result = await uploadDocument(file);
      
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
  }, [onUploadSuccess, onUploadError]);

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

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-4">
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