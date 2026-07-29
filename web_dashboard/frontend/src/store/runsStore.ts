/**
 * Zustand store for run state management
 */
import { create } from 'zustand';
import type { RunStatus, RunSummary } from '../types/api';

interface CurrentRun {
  runId: string;
  status: RunStatus | null;
  isPolling: boolean;
  error: string | null;
}

interface RunsStore {
  // Current run being viewed/worked on
  currentRun: CurrentRun | null;
  setCurrentRun: (run: CurrentRun | null) => void;
  updateCurrentStatus: (status: RunStatus) => void;
  setPolling: (isPolling: boolean) => void;

  // Runs list
  runs: RunSummary[];
  setRuns: (runs: RunSummary[]) => void;
  refreshRuns: () => Promise<void>;

  // Upload state
  uploadProgress: number;
  setUploadProgress: (progress: number) => void;

  // Global error
  error: string | null;
  setError: (error: string | null) => void;
}

export const useRunsStore = create<RunsStore>((set) => ({
  currentRun: null,
  runs: [],
  uploadProgress: 0,
  error: null,

  setCurrentRun: (run) => set({ currentRun: run }),

  // Upsert keyed by run_id: a status for a run we are not currently holding
  // (deep link, refresh, click-through from the Dashboard) replaces the
  // current run instead of being silently dropped or merged into the wrong
  // run's record.
  updateCurrentStatus: (status) =>
    set((state) => ({
      currentRun:
        state.currentRun && state.currentRun.runId === status.run_id
          ? { ...state.currentRun, status }
          : { runId: status.run_id, status, isPolling: false, error: null },
    })),

  setPolling: (isPolling) =>
    set((state) => ({
      currentRun: state.currentRun
        ? { ...state.currentRun, isPolling }
        : null,
    })),

  setRuns: (runs) => set({ runs }),

  refreshRuns: async () => {
    const { runsApi } = await import('../api/runs');
    try {
      const response = await runsApi.list();
      set({ runs: response.runs });
    } catch (error) {
      console.error('Failed to refresh runs:', error);
    }
  },

  setUploadProgress: (progress) => set({ uploadProgress: progress }),

  setError: (error) => set({ error }),
}));
