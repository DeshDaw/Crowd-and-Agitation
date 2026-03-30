/**
 * Run progress monitoring component
 */
import { useEffect, useState } from 'react';
import { Activity, Clock, Cpu, FileImage, CheckCircle, XCircle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { ProgressBar } from '../ui/ProgressBar';
import { StatusBadge } from '../ui/StatusBadge';
import { Button } from '../ui/Button';
import type { RunStatus } from '../../types/api';

interface ProgressMonitorProps {
  status: RunStatus;
  onCancel: () => void;
}

export const ProgressMonitor = ({ status, onCancel }: ProgressMonitorProps) => {
  const [elapsed, setElapsed] = useState(0);

  const progress = status.progress;
  const percentComplete = progress.total_frames > 0
    ? (progress.processed_frames / progress.total_frames) * 100
    : 0;

  // Calculate elapsed time
  useEffect(() => {
    if (!status.started_at) return;

    const interval = setInterval(() => {
      const start = new Date(status.started_at!).getTime();
      const now = Date.now();
      setElapsed(Math.floor((now - start) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [status.started_at]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const isRunning = status.state === 'processing';
  const isFinished = ['completed', 'failed', 'cancelled'].includes(status.state);

  return (
    <Card className={`${isFinished ? 'border-l-4' : ''} ${status.state === 'completed' ? 'border-l-green-500' : status.state === 'failed' ? 'border-l-red-500' : 'border-l-amber-500'}`}
    >
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CardTitle>Run Progress</CardTitle>
            <StatusBadge state={status.state} />
          </div>
          {isRunning && (
            <Button variant="danger" size="sm" onClick={onCancel}>
              Cancel
            </Button>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Status Icon */}
        {status.state === 'completed' && (
          <div className="flex items-center gap-3 text-green-600">
            <CheckCircle className="h-8 w-8" />
            <span className="font-medium">Processing completed successfully!</span>
          </div>
        )}
        {status.state === 'failed' && (
          <div className="flex items-center gap-3 text-red-600">
            <XCircle className="h-8 w-8" />
            <div>
              <span className="font-medium">Processing failed</span>
              {status.error_message && (
                <p className="text-sm text-red-500 mt-1">{status.error_message}</p>
              )}
            </div>
          </div>
        )}
        {status.state === 'cancelled' && (
          <div className="flex items-center gap-3 text-amber-600">
            <XCircle className="h-8 w-8" />
            <span className="font-medium">Run cancelled by user</span>
          </div>
        )}

        {/* Progress Bar */}
        <div>
          <div className="flex justify-between text-sm text-slate-600 mb-2">
            <span>
              {isRunning ? 'Processing...' : status.state === 'completed' ? 'Complete' : 'Ready'}
            </span>
            {progress.total_frames > 0 && (
              <span>
                {progress.processed_frames} / {progress.total_frames} frames
              </span>
            )}
          </div>
          <ProgressBar
            progress={percentComplete}
            current={progress.processed_frames}
            total={progress.total_frames}
            variant={status.state === 'failed' ? 'error' : 'default'}
          />
        </div>

        {/* Current Frame */}
        {progress.current_frame && (
          <div className="flex items-center gap-2 p-3 bg-slate-50 rounded-md">
            <FileImage className="h-5 w-5 text-slate-400" />
            <span className="text-sm text-slate-600">
              Current: <span className="font-medium text-slate-900">{progress.current_frame}</span>
            </span>
          </div>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-3 bg-slate-50 rounded-md">
            <div className="flex items-center gap-2 text-slate-500 mb-1">
              <Cpu className="h-4 w-4" />
              <span className="text-xs">Device</span>
            </div>
            <span className="text-sm font-medium capitalize">{status.config.device}</span>
          </div>

          <div className="p-3 bg-slate-50 rounded-md">
            <div className="flex items-center gap-2 text-slate-500 mb-1">
              <Activity className="h-4 w-4" />
              <span className="text-xs">Stage</span>
            </div>
            <span className="text-sm font-medium capitalize">
              {progress.current_stage || 'Idle'}
            </span>
          </div>

          <div className="p-3 bg-slate-50 rounded-md">
            <div className="flex items-center gap-2 text-slate-500 mb-1">
              <Clock className="h-4 w-4" />
              <span className="text-xs">Elapsed</span>
            </div>
            <span className="text-sm font-medium">{formatTime(elapsed)}</span>
          </div>

          {progress.per_stage_timings.avg_detection_ms && (
            <div className="p-3 bg-slate-50 rounded-md">
              <div className="flex items-center gap-2 text-slate-500 mb-1">
                <Clock className="h-4 w-4" />
                <span className="text-xs">Avg Inference</span>
              </div>
              <span className="text-sm font-medium">
                {Math.round(progress.per_stage_timings.avg_detection_ms)}ms
              </span>
            </div>
          )}
        </div>

        {/* Per-stage timings */}
        {(progress.per_stage_timings.avg_detection_ms || progress.per_stage_timings.avg_pose_ms) && (
          <div className="border-t pt-4">
            <h4 className="text-sm font-medium text-slate-900 mb-2">Average Inference Times</h4>
            <div className="space-y-1">
              {progress.per_stage_timings.avg_detection_ms && (
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">Detection</span>
                  <span className="font-medium">
                    {Math.round(progress.per_stage_timings.avg_detection_ms)}ms
                  </span>
                </div>
              )}
              {progress.per_stage_timings.avg_pose_ms && (
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">Pose</span>
                  <span className="font-medium">
                    {Math.round(progress.per_stage_timings.avg_pose_ms)}ms
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
