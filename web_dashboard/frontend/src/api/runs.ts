/**
 * API calls for run management
 */
import { apiClient, API_BASE_URL } from './client';
import type {
  EscalationEvent,
  FrameMetrics,
  HealthResponse,
  RunConfig,
  RunCreateResponse,
  RunListResponse,
  RunStatus,
  SummaryStats,
} from '../types/api';

export interface CreateRunRequest {
  config?: Partial<RunConfig>;
}

export const runsApi = {
  // Health check
  health: async (): Promise<HealthResponse> => {
    const response = await apiClient.get<HealthResponse>('/health');
    return response.data;
  },

  // Create a new run
  create: async (data?: CreateRunRequest): Promise<RunCreateResponse> => {
    const formData = new FormData();
    if (data?.config) {
      formData.append('config', JSON.stringify(data.config));
    }
    const response = await apiClient.post<RunCreateResponse>('/runs', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  // Upload files to a run
  uploadFiles: async (
    runId: string,
    files: File[],
    video?: File,
    onProgress?: (percent: number) => void
  ): Promise<{ run_id: string; uploaded_files: string[]; total_files: number }> => {
    const formData = new FormData();

    if (files.length > 0) {
      files.forEach((file) => formData.append('files', file));
    }

    if (video) {
      formData.append('video', video);
    }

    // No timeout: large image batches / videos legitimately exceed the
    // shared 60s client timeout on slow connections.
    const response = await apiClient.post(`/runs/${runId}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 0,
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.min(99, Math.round((e.loaded / e.total) * 100)));
        }
      },
    });
    return response.data;
  },

  // Start processing; the user's final config edits ride along so the run
  // executes with what the Configure step shows, not creation-time defaults.
  start: async (
    runId: string,
    config?: Partial<RunConfig>
  ): Promise<{ run_id: string; status: string; message: string }> => {
    const response = await apiClient.post(`/runs/${runId}/start`, config ?? null);
    return response.data;
  },

  // Get run status
  getStatus: async (runId: string): Promise<RunStatus> => {
    const response = await apiClient.get<RunStatus>(`/runs/${runId}/status`);
    return response.data;
  },

  // Get summary
  getSummary: async (runId: string): Promise<SummaryStats> => {
    const response = await apiClient.get<SummaryStats>(`/runs/${runId}/summary`);
    return response.data;
  },

  // Get metrics
  getMetrics: async (runId: string): Promise<FrameMetrics[]> => {
    const response = await apiClient.get<FrameMetrics[]>(`/runs/${runId}/metrics`);
    return response.data;
  },

  // Get events
  getEvents: async (runId: string): Promise<{ events: EscalationEvent[]; total_events: number }> => {
    const response = await apiClient.get(`/runs/${runId}/events`);
    return response.data;
  },

  // List all runs
  list: async (): Promise<RunListResponse> => {
    const response = await apiClient.get<RunListResponse>('/runs');
    return response.data;
  },

  // Cancel run
  cancel: async (runId: string): Promise<{ run_id: string; status: string; message: string }> => {
    const response = await apiClient.post(`/runs/${runId}/cancel`);
    return response.data;
  },

  // Delete run
  delete: async (runId: string): Promise<{ run_id: string; deleted: boolean }> => {
    const response = await apiClient.delete(`/runs/${runId}`);
    return response.data;
  },

  // Get files
  getFiles: async (runId: string): Promise<{
    files: { name: string; path: string; type: string; size_bytes: number; modified_at: string }[];
    categories: Record<string, any[]>;
  }> => {
    const response = await apiClient.get(`/runs/${runId}/files`);
    return response.data;
  },

  // Download file
  download: (runId: string, path: string): string => {
    return `${API_BASE_URL}/api/runs/${runId}/download?path=${encodeURIComponent(path)}`;
  },

  // Get artifact URL (named artifacts require their extensioned route names,
  // e.g. "summary.json", "density_plot.png" — matching files.py)
  getArtifactUrl: (runId: string, artifact: string, filename?: string): string => {
    if (filename && ['annotated', 'heatmaps', 'escalation'].includes(artifact)) {
      return `/api/runs/${runId}/artifacts/${artifact}/${encodeURIComponent(filename)}`;
    }
    return `/api/runs/${runId}/artifacts/${artifact}`;
  },

  // Calibration
  saveCalibration: async (
    runId: string,
    data: { image_points: number[][]; width_m: number; height_m: number; image_size: number[] }
  ): Promise<{ run_id: string; saved: boolean; area_m2: number }> => {
    const response = await apiClient.post(`/runs/${runId}/calibration`, data);
    return response.data;
  },

  getCalibration: async (
    runId: string
  ): Promise<{ image_points: number[][]; width_m: number; height_m: number; image_size: number[] }> => {
    const response = await apiClient.get(`/runs/${runId}/calibration`);
    return response.data;
  },

  // List frames
  listAnnotated: async (runId: string): Promise<string[]> => {
    const response = await apiClient.get<string[]>(`/runs/${runId}/artifacts/annotated`);
    return response.data;
  },

  listHeatmaps: async (runId: string): Promise<string[]> => {
    const response = await apiClient.get<string[]>(`/runs/${runId}/artifacts/heatmaps`);
    return response.data;
  },

  listEscalation: async (runId: string): Promise<{
    filename: string;
    frame_index: number;
    agitation_score: number;
    density_ratio: number;
  }[]> => {
    const response = await apiClient.get(`/runs/${runId}/artifacts/escalation`);
    return response.data;
  },
};

// Export API_BASE_URL for use in other modules
export { API_BASE_URL };
