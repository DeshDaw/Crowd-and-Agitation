/**
 * Card component
 */
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  noPadding?: boolean;
}

export const Card = ({ children, className, noPadding = false }: CardProps) => {
  return (
    <div
      className={twMerge(
        clsx(
          'bg-white rounded-lg border border-slate-200 shadow-sm',
          !noPadding && 'p-6',
          className
        )
      )}
    >
      {children}
    </div>
  );
};

export const CardHeader = ({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) => {
  return (
    <div className={twMerge('mb-4', className)}>
      {children}
    </div>
  );
};

export const CardTitle = ({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) => {
  return (
    <h3 className={twMerge('text-lg font-semibold text-slate-900', className)}>
      {children}
    </h3>
  );
};

export const CardDescription = ({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) => {
  return (
    <p className={twMerge('text-sm text-slate-500 mt-1', className)}>
      {children}
    </p>
  );
};

export const CardContent = ({
  children,
  className,
  noPadding = false,
}: {
  children: ReactNode;
  className?: string;
  noPadding?: boolean;
}) => {
  // noPadding gives full-bleed content (tables, divided lists) by negating
  // the parent Card's p-6 inset
  return (
    <div className={twMerge(clsx(noPadding && '-mx-6 -mb-6', className))}>
      {children}
    </div>
  );
};
