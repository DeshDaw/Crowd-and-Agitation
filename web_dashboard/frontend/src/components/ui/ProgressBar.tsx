/**
 * Progress bar component
 */
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface ProgressBarProps {
  progress: number;
  total?: number;
  current?: number;
  showText?: boolean;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'default' | 'success' | 'error';
  className?: string;
}

export const ProgressBar = ({
  progress,
  total,
  current,
  showText = true,
  size = 'md',
  variant = 'default',
  className,
}: ProgressBarProps) => {
  const percentage = Math.min(100, Math.max(0, progress));

  const sizeStyles = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4',
  };

  const variantStyles = {
    default: 'bg-primary-600',
    success: 'bg-green-600',
    error: 'bg-red-600',
  };

  return (
    <div className={twMerge('w-full', className)}>
      <div
        className={twMerge(
          clsx(
            'w-full bg-slate-200 rounded-full overflow-hidden',
            sizeStyles[size]
          )
        )}
      >
        <div
          className={twMerge(
            clsx(
              'transition-all duration-300 ease-out rounded-full',
              sizeStyles[size],
              variantStyles[variant]
            )
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showText && (
        <div className="mt-2 flex justify-between text-sm text-slate-600">
          <span>{percentage.toFixed(1)}%</span>
          {current !== undefined && total !== undefined && (
            <span>
              {current} / {total} frames
            </span>
          )}
        </div>
      )}
    </div>
  );
};
