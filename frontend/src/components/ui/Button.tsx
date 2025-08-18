import React from 'react';
import { motion } from 'framer-motion';
import { ButtonProps } from '../../types';
import { cn } from '../../utils/cn';
import { Loader2 } from 'lucide-react';

const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  children,
  onClick,
  disabled = false,
  loading = false,
  className,
  ...props
}) => {
  const baseClasses = 'inline-flex items-center justify-center gap-2 font-medium rounded-xl transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';
  
  const variants = {
    primary: 'bg-gradient-to-r from-primary-600 to-primary-700 text-white shadow-soft hover:shadow-medium hover:scale-105 focus:ring-primary-500 active:scale-95',
    secondary: 'bg-white text-slate-700 border border-slate-200 shadow-soft hover:shadow-medium hover:border-slate-300 hover:scale-105 focus:ring-slate-500 active:scale-95',
    ghost: 'bg-transparent text-slate-600 hover:bg-slate-100 hover:text-slate-900 focus:ring-slate-500',
    danger: 'bg-gradient-to-r from-error-600 to-error-700 text-white shadow-soft hover:shadow-medium hover:scale-105 focus:ring-error-500 active:scale-95',
    success: 'bg-gradient-to-r from-success-600 to-success-700 text-white shadow-soft hover:shadow-medium hover:scale-105 focus:ring-success-500 active:scale-95',
  };

  const sizes = {
    sm: 'px-3 py-2 text-sm',
    md: 'px-4 py-2.5 text-sm',
    lg: 'px-6 py-3 text-base',
  };

  const classes = cn(
    baseClasses,
    variants[variant],
    sizes[size],
    className
  );

  return (
    <motion.button
      className={classes}
      onClick={onClick}
      disabled={disabled || loading}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      {...props}
    >
      {loading && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
        >
          <Loader2 className="h-4 w-4 animate-spin" />
        </motion.div>
      )}
      <motion.span
        initial={{ opacity: loading ? 0.5 : 1 }}
        animate={{ opacity: loading ? 0.5 : 1 }}
      >
        {children}
      </motion.span>
    </motion.button>
  );
};

export default Button;