/**
 * Run creation page
 */
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Play, AlertCircle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { FileUploader } from '../components/upload/FileUploader';
import { ConfigForm } from '../components/upload/ConfigForm';
import { useRun } from '../hooks/useRun';
import { useRunsStore } from '../store/runsStore';
import { runsApi } from '../api/runs';
import type { RunConfig } from '../types/api';

export const RunCreate = () => {
  const navigate = useNavigate();
  const { createRun, uploadFiles, startProcessing, resetUploadProgress } = useRun();
  const uploadProgress = useRunsStore((s) => s.uploadProgress);

  const [step, setStep] = useState<'upload' | 'config' | 'processing'>('upload');
  const [runId, setRunId] = useState<string | null>(null);
  const [config, setConfig] = useState<Partial<RunConfig>>({
    device: 'cpu',
    confidence_threshold: 0.5,
    pose_confidence_threshold: 0.5,
    max_inference_width: 960,
    tracker_iou_threshold: 0.3,
    tracker_max_lost: 5,
    density_low_sigma: 0.5,
    density_high_sigma: 1.5,
    agitation_threshold_sigma: 2.0,
    save_annotated: true,
    save_heatmaps: true,
    generate_plots: true,
    save_database: true,
  });
  const [cudaAvailable, setCudaAvailable] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Check CUDA availability
  useEffect(() => {
    runsApi.health().then((h) => setCudaAvailable(h.cuda_available)).catch(() => {});
  }, []);

  // Entering the upload step must always leave it usable (progress reset)
  useEffect(() => {
    if (step === 'upload') resetUploadProgress();
  }, [step, resetUploadProgress]);

  const handleFilesSelected = async (files: File[], video?: File) => {
    let newRunId: string | null = null;
    try {
      setError(null);
      newRunId = await createRun(config);
      setRunId(newRunId);
      await uploadFiles(newRunId, files, video);
      setStep('config');
    } catch (err: any) {
      // Delete the orphan run so failed attempts don't pile up as
      // permanently-"created" entries in the dashboard list
      if (newRunId) {
        runsApi.delete(newRunId).catch(() => {});
        setRunId(null);
      }
      setError(err.message || 'Failed to upload files');
    }
  };

  const handleStart = async () => {
    if (!runId) return;
    try {
      setError(null);
      // Send the user's final config — edits made in the Configure step
      // after upload must reach the run
      await startProcessing(runId, config);
      navigate(`/runs/${runId}`);
    } catch (err: any) {
      setError(err.message || 'Failed to start processing');
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              className="p-2 hover:bg-slate-100 rounded"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
            <h1 className="text-xl font-bold">New Processing Run</h1>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Steps */}
        <div className="flex items-center gap-4 mb-8">
          {['Upload Files', 'Configure', 'Process'].map((label, idx) => (
            <div key={label} className="flex items-center gap-2">
              <div className={`
                w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium
                ${idx === 0 ? 'bg-green-100 text-green-700' :
                  idx === 1 && step === 'config' ? 'bg-primary-100 text-primary-700' :
                  'bg-slate-100 text-slate-500'
                }
              `}>
                {idx + 1}
              </div>
              <span className={`text-sm ${idx < (step === 'upload' ? 0 : step === 'config' ? 1 : 2) ? 'text-slate-900' : 'text-slate-500'}`}>
                {label}
              </span>
              {idx < 2 && <span className="text-slate-300">→</span>}
            </div>
          ))}
        </div>

        {error && (
          <div className="mb-6 flex items-center gap-2 p-4 bg-red-50 text-red-700 rounded-lg">
            <AlertCircle className="h-5 w-5" />
            <span>{error}</span>
          </div>
        )}

        {step === 'upload' && (
          <Card>
            <CardHeader>
              <CardTitle>Upload Images or Video</CardTitle>
            </CardHeader>
            <CardContent>
              <FileUploader
                onFilesSelected={handleFilesSelected}
                uploadProgress={uploadProgress}
              />
            </CardContent>
          </Card>
        )}

        {step === 'config' && (
          <div className="space-y-6">
            <ConfigForm
              config={config}
              onChange={setConfig}
              cudaAvailable={cudaAvailable}
            />

            <div className="flex justify-end gap-4">
              <Button variant="secondary" onClick={() => setStep('upload')}>
                Back
              </Button>
              <Button onClick={handleStart} size="lg">
                <Play className="h-5 w-5 mr-2" />
                Start Processing
              </Button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};
