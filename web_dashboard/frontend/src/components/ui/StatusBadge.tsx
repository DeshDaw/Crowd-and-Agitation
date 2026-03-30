/**
 * Status badge for run states
 */
import { Badge } from './Badge';
import type { RunState } from '../../types/api';

interface StatusBadgeProps {
  state: RunState;
}

export const StatusBadge = ({ state }: StatusBadgeProps) => {
  const variantMap: Record<RunState, 'default' | 'success' | 'warning' | 'error' | 'info'> = {
    created: 'default',
    uploading: 'info',
    queued: 'info',
    processing: 'info',
    completed: 'success',
    failed: 'error',
    cancelled: 'warning',
  };

  const labelMap: Record<RunState, string> = {
    created: 'Created',
    uploading: 'Uploading',
    queued: 'Queued',
    processing: 'Processing',
    completed: 'Completed',
    failed: 'Failed',
    cancelled: 'Cancelled',
  };

  return <Badge variant={variantMap[state]}>{labelMap[state]}</Badge>;
};
