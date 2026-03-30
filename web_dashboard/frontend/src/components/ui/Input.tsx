/**
 * Input and Label components
 */
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { InputHTMLAttributes, ReactNode } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
}

export const Input = ({
  label,
  error,
  helperText,
  className,
  id,
  ...props
}: InputProps) => {
  const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;

  return (
    <div className="w-full">
      {label && (
        <label
          htmlFor={inputId}
          className="block text-sm font-medium text-slate-700 mb-1"
        >
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={twMerge(
          clsx(
            'block w-full rounded-md border-slate-300 shadow-sm',
            'focus:border-primary-500 focus:ring-primary-500 sm:text-sm',
            'disabled:bg-slate-100 disabled:cursor-not-allowed',
            error && 'border-red-300 focus:border-red-500 focus:ring-red-500',
            className
          )
        )}
        {...props}
      />
      {error && (
        <p className="mt-1 text-sm text-red-600">{error}</p>
      )}
      {helperText && !error && (
        <p className="mt-1 text-sm text-slate-500">{helperText}</p>
      )}
    </div>
  );
};

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
  options: { value: string; label: string }[];
}

export const Select = ({
  label,
  error,
  options,
  className,
  id,
  ...props
}: SelectProps) => {
  const selectId = id || `select-${Math.random().toString(36).substr(2, 9)}`;

  return (
    <div className="w-full">
      {label && (
        <label
          htmlFor={selectId}
          className="block text-sm font-medium text-slate-700 mb-1"
        >
          {label}
        </label>
      )}
      <select
        id={selectId}
        className={twMerge(
          clsx(
            'block w-full rounded-md border-slate-300 shadow-sm',
            'focus:border-primary-500 focus:ring-primary-500 sm:text-sm',
            error && 'border-red-300 focus:border-red-500 focus:ring-red-500',
            className
          )
        )}
        {...props}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {error && (
        <p className="mt-1 text-sm text-red-600">{error}</p>
      )}
    </div>
  );
};

interface CheckboxProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: ReactNode;
}

export const Checkbox = ({ label, className, ...props }: CheckboxProps) => {
  return (
    <div className={twMerge('flex items-center', className)}>
      <input
        type="checkbox"
        className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-slate-300 rounded"
        {...props}
      />
      {label && (
        <label className="ml-2 block text-sm text-slate-900">{label}</label>
      )}
    </div>
  );
};
