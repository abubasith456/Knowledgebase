import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  Upload, 
  Search, 
  FileText, 
  Settings, 
  Brain,
  Sparkles,
  Zap,
  Database,
  BarChart3,
  Menu
} from 'lucide-react';
import { SidebarProps, NavItem } from '../types';
import Button from './ui/Button';
import { cn } from '../utils/cn';

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  const navItems: NavItem[] = [
    {
      id: 'upload',
      name: 'Upload Documents',
      icon: Upload,
      description: 'Upload and process your documents',
    },
    {
      id: 'search',
      name: 'Search Knowledge Base',
      icon: Search,
      description: 'Query your documents with AI',
    },
    {
      id: 'documents',
      name: 'Manage Documents',
      icon: FileText,
      description: 'View and manage your documents',
    },
    {
      id: 'analytics',
      name: 'Analytics',
      icon: BarChart3,
      description: 'View system statistics and insights',
    },
    {
      id: 'settings',
      name: 'Settings',
      icon: Settings,
      description: 'Configure your preferences',
    },
  ];

  return (
    <>
      {/* Mobile Overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="sidebar-overlay"
            onClick={onClose}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        className={cn(
          'sidebar',
          !isOpen && 'sidebar-collapsed'
        )}
        initial={{ x: -256 }}
        animate={{ x: isOpen ? 0 : -256 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/20">
          <div className="flex items-center space-x-3">
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
              <h1 className="text-lg font-bold gradient-text">Knowledge Base</h1>
              <p className="text-xs text-slate-500">AI-powered intelligence</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="lg:hidden"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {navItems.map((item, index) => {
            const Icon = item.icon;
            return (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <button
                  className="sidebar-nav-item w-full text-left group"
                  onClick={() => {
                    // Handle navigation here
                    console.log(`Navigating to ${item.id}`);
                  }}
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-2 bg-slate-100 group-hover:bg-primary-100 rounded-lg transition-colors duration-200">
                      <Icon className="h-4 w-4 text-slate-600 group-hover:text-primary-600 transition-colors duration-200" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium text-slate-900 group-hover:text-primary-700 transition-colors duration-200">
                        {item.name}
                      </p>
                      <p className="text-xs text-slate-500 group-hover:text-slate-600 transition-colors duration-200">
                        {item.description}
                      </p>
                    </div>
                  </div>
                </button>
              </motion.div>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-white/20">
          <div className="space-y-3">
            {/* Quick Stats */}
            <div className="bg-gradient-to-r from-primary-50 to-accent-50 rounded-xl p-3">
              <div className="flex items-center space-x-2 mb-2">
                <Database className="h-4 w-4 text-primary-600" />
                <span className="text-sm font-medium text-slate-900">System Status</span>
              </div>
              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-600">Documents</span>
                  <span className="font-medium text-slate-900">12</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-600">Chunks</span>
                  <span className="font-medium text-slate-900">1,247</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-600">Status</span>
                  <span className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-success-500 rounded-full"></div>
                    <span className="text-success-600 font-medium">Online</span>
                  </span>
                </div>
              </div>
            </div>

            {/* Version Info */}
            <div className="text-center">
              <div className="flex items-center justify-center space-x-1 text-xs text-slate-500">
                <Sparkles className="h-3 w-3" />
                <span>v2.1.0</span>
                <span>•</span>
                <span>Powered by AI</span>
              </div>
            </div>
          </div>
        </div>
      </motion.aside>
    </>
  );
};

export default Sidebar;