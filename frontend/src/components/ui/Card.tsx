import React from 'react';
import { motion } from 'framer-motion';
import { CardProps } from '../../types';
import { cn } from '../../utils/cn';

const Card: React.FC<CardProps> = ({
  variant = 'default',
  children,
  className,
  onClick,
  ...props
}) => {
  const baseClasses = 'rounded-2xl transition-all duration-300';
  
  const variants = {
    default: 'bg-white/70 backdrop-blur-sm border border-white/20 shadow-soft hover:shadow-medium p-6',
    glass: 'bg-white/10 backdrop-blur-md border border-white/20 shadow-soft p-6',
    gradient: 'bg-gradient-to-br from-white/80 to-white/40 backdrop-blur-sm border border-white/30 shadow-medium p-6',
  };

  const classes = cn(
    baseClasses,
    variants[variant],
    onClick && 'cursor-pointer hover:scale-[1.02]',
    className
  );

  return (
    <motion.div
      className={classes}
      onClick={onClick}
      whileHover={onClick ? { scale: 1.02, y: -2 } : undefined}
      whileTap={onClick ? { scale: 0.98 } : undefined}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      {...props}
    >
      {children}
    </motion.div>
  );
};

export default Card;