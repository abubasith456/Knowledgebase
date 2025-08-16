import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import { 
  Upload, 
  Search, 
  FileText, 
  Brain,
  Github,
  ExternalLink,
  Sparkles,
  Zap,
  Database
} from 'lucide-react';
import { AppProvider, useApp } from './context/AppContext';
import FileUpload from './components/FileUpload';
import QueryInterface from './components/QueryInterface';
import DocumentManager from './components/DocumentManager';
import Button from './components/ui/Button';
import Badge from './components/ui/Badge';

const AppContent: React.FC = () => {
  const { uiState, setUIState, stats } = useApp();

  const tabs = [
    {
      id: 'upload',
      name: 'Upload Documents',
      icon: Upload,
      description: 'Upload and process your documents',
      component: <FileUpload />,
    },
    {
      id: 'search',
      name: 'Search Knowledge Base',
      icon: Search,
      description: 'Query your documents with AI',
      component: <QueryInterface />,
    },
    {
      id: 'documents',
      name: 'Manage Documents',
      icon: FileText,
      description: 'View and manage your documents',
      component: <DocumentManager onDocumentDeleted={() => {}} />,
    },
  ];

  const currentTab = tabs.find(tab => tab.id === uiState.activeTab);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '12px',
            color: '#1e293b',
            boxShadow: '0 10px 40px -10px rgba(0, 0, 0, 0.15)',
          },
          success: {
            iconTheme: {
              primary: '#22c55e',
              secondary: '#fff',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />

      {/* Header */}
      <motion.header 
        className="sticky top-0 z-40 bg-white/80 backdrop-blur-md border-b border-white/20 shadow-soft"
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        <div className="container-modern">
          <div className="flex items-center justify-between h-16">
            <motion.div 
              className="flex items-center space-x-4"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="relative">
                <div className="p-2 bg-gradient-to-r from-primary-600 to-accent-600 rounded-xl shadow-glow">
                  <Brain className="h-6 w-6 text-white" />
                </div>
                <motion.div
                  className="absolute -top-1 -right-1 w-3 h-3 bg-success-500 rounded-full"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </div>
              <div>
                <h1 className="text-xl font-bold gradient-text">Knowledge Base</h1>
                <p className="text-sm text-slate-500">AI-powered document intelligence</p>
              </div>
            </motion.div>
            
            <motion.div 
              className="flex items-center space-x-4"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              {stats && (
                <div className="hidden md:flex items-center space-x-3">
                  <Badge variant="primary" className="text-xs">
                    <Database className="h-3 w-3" />
                    {stats.total_chunks} chunks
                  </Badge>
                  <Badge variant="success" className="text-xs">
                    <FileText className="h-3 w-3" />
                    {stats.unique_documents} docs
                  </Badge>
                </div>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => window.open('http://localhost:8000/docs', '_blank')}
              >
                <ExternalLink className="h-4 w-4" />
                <span className="hidden sm:inline">API Docs</span>
              </Button>
            </motion.div>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <main className="container-modern section-padding">
        {/* Hero Section */}
        <motion.div 
          className="text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h2 className="text-4xl md:text-5xl font-bold gradient-text mb-4">
            Intelligent Document Management
          </h2>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto">
            Upload, process, and search through your documents with AI-powered semantic understanding
          </p>
        </motion.div>

        {/* Tab Navigation */}
        <motion.div 
          className="mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex flex-wrap justify-center gap-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = uiState.activeTab === tab.id;
              
              return (
                <motion.button
                  key={tab.id}
                  onClick={() => setUIState({ activeTab: tab.id })}
                  className={`
                    group relative px-6 py-3 rounded-xl font-medium transition-all duration-300
                    ${isActive
                      ? 'bg-white/80 backdrop-blur-sm text-primary-700 shadow-medium'
                      : 'bg-white/40 backdrop-blur-sm text-slate-600 hover:bg-white/60 hover:text-slate-800'
                    }
                  `}
                  whileHover={{ scale: 1.05, y: -2 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <div className="flex items-center space-x-2">
                    <Icon className={`h-5 w-5 transition-colors ${isActive ? 'text-primary-600' : 'text-slate-500 group-hover:text-slate-700'}`} />
                    <span>{tab.name}</span>
                  </div>
                  {isActive && (
                    <motion.div
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-primary-600 to-accent-600 rounded-full"
                      layoutId="activeTab"
                      initial={false}
                      transition={{ type: "spring", stiffness: 500, damping: 30 }}
                    />
                  )}
                </motion.button>
              );
            })}
          </div>
        </motion.div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={uiState.activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="space-y-6"
          >
            {currentTab?.component}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Footer */}
      <motion.footer 
        className="bg-white/50 backdrop-blur-sm border-t border-white/20 mt-20"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
      >
        <div className="container-modern py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="md:col-span-2">
              <div className="flex items-center space-x-2 mb-4">
                <div className="p-2 bg-gradient-to-r from-primary-600 to-accent-600 rounded-lg">
                  <Sparkles className="h-5 w-5 text-white" />
                </div>
                <h3 className="text-lg font-bold gradient-text">Knowledge Base System</h3>
              </div>
              <p className="text-slate-600 mb-4">
                A modern, AI-powered knowledge base system that transforms how you manage and search through documents.
                Built with cutting-edge technology for optimal performance and user experience.
              </p>
              <div className="flex items-center space-x-4">
                <Button variant="ghost" size="sm">
                  <Github className="h-4 w-4" />
                  GitHub
                </Button>
                <Button variant="ghost" size="sm">
                  <ExternalLink className="h-4 w-4" />
                  Documentation
                </Button>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold text-slate-900 mb-4">Technology</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li className="flex items-center space-x-2">
                  <Zap className="h-4 w-4 text-primary-600" />
                  <span>FastAPI Backend</span>
                </li>
                <li className="flex items-center space-x-2">
                  <Zap className="h-4 w-4 text-primary-600" />
                  <span>React + TypeScript</span>
                </li>
                <li className="flex items-center space-x-2">
                  <Zap className="h-4 w-4 text-primary-600" />
                  <span>ChromaDB Vector DB</span>
                </li>
                <li className="flex items-center space-x-2">
                  <Zap className="h-4 w-4 text-primary-600" />
                  <span>Jina Embeddings</span>
                </li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold text-slate-900 mb-4">Features</h4>
              <ul className="space-y-2 text-sm text-slate-600">
                <li>• Smart Document Processing</li>
                <li>• Semantic Search</li>
                <li>• Multiple Indexing Modes</li>
                <li>• Real-time Updates</li>
                <li>• Modern UI/UX</li>
                <li>• REST API</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-8 pt-8 border-t border-slate-200">
            <p className="text-center text-sm text-slate-500">
              © 2024 Knowledge Base System. Built with modern web technologies and AI.
            </p>
          </div>
        </div>
      </motion.footer>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
};

export default App;