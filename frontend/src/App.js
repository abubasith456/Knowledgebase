import React, { useState } from 'react';
import { Toaster } from 'react-hot-toast';
import { 
  Upload, 
  Search, 
  FileText, 
  Settings, 
  Brain,
  Github,
  ExternalLink
} from 'lucide-react';
import FileUpload from './components/FileUpload';
import QueryInterface from './components/QueryInterface';
import DocumentManager from './components/DocumentManager';

const App = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleUploadSuccess = () => {
    // Trigger refresh of document list
    setRefreshTrigger(prev => prev + 1);
  };

  const handleDocumentDeleted = () => {
    // Trigger refresh of document list
    setRefreshTrigger(prev => prev + 1);
  };

  const tabs = [
    {
      id: 'upload',
      name: 'Upload Documents',
      icon: Upload,
      component: (
        <FileUpload 
          onUploadSuccess={handleUploadSuccess}
          onUploadError={() => {}}
        />
      ),
    },
    {
      id: 'search',
      name: 'Search Knowledge Base',
      icon: Search,
      component: <QueryInterface />,
    },
    {
      id: 'documents',
      name: 'Manage Documents',
      icon: FileText,
      component: (
        <DocumentManager 
          key={refreshTrigger}
          onDocumentDeleted={handleDocumentDeleted}
        />
      ),
    },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: 3000,
            iconTheme: {
              primary: '#10B981',
              secondary: '#fff',
            },
          },
          error: {
            duration: 5000,
            iconTheme: {
              primary: '#EF4444',
              secondary: '#fff',
            },
          },
        }}
      />

      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-primary-600 to-primary-700 rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Knowledge Base System</h1>
                <p className="text-sm text-gray-500">AI-powered document search and management</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 text-sm text-gray-600 hover:text-primary-600 transition-colors"
              >
                <ExternalLink className="h-4 w-4" />
                <span>API Docs</span>
              </a>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
              >
                <Github className="h-4 w-4" />
                <span>GitHub</span>
              </a>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation */}
        <div className="mb-8">
          <nav className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`
                    flex items-center space-x-2 px-3 py-2 text-sm font-medium rounded-md transition-colors
                    ${isActive
                      ? 'bg-primary-100 text-primary-700 border-b-2 border-primary-600'
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                    }
                  `}
                >
                  <Icon className="h-4 w-4" />
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="animate-fade-in">
          {tabs.find(tab => tab.id === activeTab)?.component}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-sm font-semibold text-gray-900 tracking-wider uppercase mb-4">
                About
              </h3>
              <p className="text-sm text-gray-600">
                A modern knowledge base system powered by AI embeddings and vector search.
                Upload documents and query them using natural language.
              </p>
            </div>
            
            <div>
              <h3 className="text-sm font-semibold text-gray-900 tracking-wider uppercase mb-4">
                Technology
              </h3>
              <ul className="text-sm text-gray-600 space-y-2">
                <li>• FastAPI Backend</li>
                <li>• React Frontend</li>
                <li>• ChromaDB Vector Database</li>
                <li>• Jina Embeddings</li>
                <li>• Tailwind CSS</li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-sm font-semibold text-gray-900 tracking-wider uppercase mb-4">
                Features
              </h3>
              <ul className="text-sm text-gray-600 space-y-2">
                <li>• Document Upload & Processing</li>
                <li>• Semantic Search</li>
                <li>• Document Management</li>
                <li>• REST API</li>
                <li>• Real-time Updates</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-8 pt-8 border-t border-gray-200">
            <p className="text-sm text-gray-500 text-center">
              © 2024 Knowledge Base System. Built with modern web technologies.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;