/**
 * Hook for managing a single run's lifecycle
 */
import { useEffect, useRef, useCallback } from 'react';
import { useRunsStore } from '../store/runsStore';
import { runsApi } from '../api/runs';
import type { RunConfig } from '../types/api';

const POLLING_INTERVAL = 2000; // 2 seconds

export const useRun = () => {
  const {
    currentRun,
    setCurrentRun,
    updateCurrentStatus,
    setPolling,
    setUploadProgress,
    setError,
  } = useRunsStore();

  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, []);

  const startPolling = useCallback(() => {
    if (!currentRun?.runId) return;

    setPolling(true);

    if (pollingRef.current) {
      clearInterval(pollingRef.current);
    }

    pollingRef.current = setInterval(async () => {
      if (!currentRun?.runId) return;

      try {
        const status = await runsApi.getStatus(currentRun.runId);
        updateCurrentStatus(status);

        // Stop polling if terminal state
        if (['completed', 'failed', 'cancelled'].includes(status.state)) {
          if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
          }
          setPolling(false);
        }
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, POLLING_INTERVAL);
  }, [currentRun?.runId, updateCurrentStatus, setPolling]);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setPolling(false);
  }, [setPolling]);

  const createRun = useCallback(
    async (config?: Partial<RunConfig>) => {
      try {
        setError(null);
        const response = await runsApi.create({ config });
        setCurrentRun({
          runId: response.run_id,
          status: null,
          isPolling: false,
          error: null,
        });
        return response.run_id;
      } catch (error: any) {
        setError(error.response?.data?.detail || 'Failed to create run');
        throw error;
      }
    },
    [setCurrentRun, setError]
  );

  const uploadFiles = useCallback(
    async (runId: string, files: File[], video?: File) => {
      try {
        setError(null);
        setUploadProgress(0);

        const response = await runsApi.uploadFiles(runId, files, video, (pct) =>
          setUploadProgress(pct)
        );

        setUploadProgress(100);
        return response;
      } catch (error: any) {
        // Reset so the upload step stays usable for a retry
        setUploadProgress(0);
        setError(error.response?.data?.detail || 'Failed to upload files');
        throw error;
      }
    },
    [setUploadProgress, setError]
  );

  const resetUploadProgress = useCallback(() => {
    setUploadProgress(0);
  }, [setUploadProgress]);

  const startProcessing = useCallback(
    async (runId: string, config?: Partial<RunConfig>) => {
      try {
        setError(null);
        const response = await runsApi.start(runId, config);
        startPolling();
        return response;
      } catch (error: any) {
        setError(error.response?.data?.detail || 'Failed to start run');
        throw error;
      }
    },
    [startPolling, setError]
  );

  const cancelRun = useCallback(
    async (runId: string) => {
      try {
        const response = await runsApi.cancel(runId);
        stopPolling();
        return response;
      } catch (error: any) {
        setError(error.response?.data?.detail || 'Failed to cancel run');
        throw error;
      }
    },
    [stopPolling, setError]
  );

  const deleteRun = useCallback(
    async (runId: string) => {
      try {
        const response = await runsApi.delete(runId);
        if (currentRun?.runId === runId) {
          setCurrentRun(null);
        }
        return response;
      } catch (error: any) {
        setError(error.response?.data?.detail || 'Failed to delete run');
        throw error;
      }
    },
    [currentRun, setCurrentRun, setError]
  );

  const fetchStatus = useCallback(
    async (runId: string) => {
      try {
        const status = await runsApi.getStatus(runId);
        updateCurrentStatus(status);
        return status;
      } catch (error: any) {
        setError(error.response?.data?.detail || 'Failed to fetch status');
        throw error;
      }
    },
    [updateCurrentStatus, setError]
  );

  const isProcessing = currentRun?.status?.state === 'processing';
  const isCompleted = currentRun?.status?.state === 'completed';
  const isFailed = currentRun?.status?.state === 'failed';
  const isPolling = currentRun?.isPolling ?? false;
  const progress = currentRun?.status?.progress ?? {
    total_frames: 0,
    processed_frames: 0,
    current_frame: null,
    current_stage: null,
    per_stage_timings: {},
  };

  return {
    runId: currentRun?.runId,
    status: currentRun?.status,
    isProcessing,
    isCompleted,
    isFailed,
    isPolling,
    progress,
    createRun,
    uploadFiles,
    resetUploadProgress,
    startProcessing,
    cancelRun,
    deleteRun,
    fetchStatus,
    startPolling,
    stopPolling,
  };
};
