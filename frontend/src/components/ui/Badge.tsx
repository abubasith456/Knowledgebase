import React from 'react';
import { motion } from 'framer-motion';
import { BadgeProps } from '../../types';
import { cn } from '../../utils/cn';

const Badge: React.FC<BadgeProps> = ({
  variant = 'primary',
  children,
  className,
  ...props
}) => {
  const baseClasses = 'inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full';
  
  const variants = {
    primary: 'bg-primary-100 text-primary-800',
    secondary: 'bg-slate-100 text-slate-700',
    success: 'bg-success-100 text-success-800',
    warning: 'bg-warning-100 text-warning-800',
    error: 'bg-error-100 text-error-800',
    accent: 'bg-accent-100 text-accent-800',
  };

  const classes = cn(
    baseClasses,
    variants[variant],
    className
  );

  return (
    <motion.span
      className={classes}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
      {...props}
    >
      {children}
    </motion.span>
  );
};

export default Badge;