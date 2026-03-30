/**
 * Run results page
 */
import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, RefreshCw } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { ProgressMonitor } from '../components/progress/ProgressMonitor';
import { SummaryCards } from '../components/results/SummaryCards';
import { ClassificationChart } from '../components/results/ClassificationChart';
import { MetricsTable } from '../components/results/MetricsTable';
import { EventsTimeline } from '../components/results/EventsTimeline';
import { PlotsDisplay } from '../components/results/PlotsDisplay';
import { Downloads } from '../components/results/Downloads';
import { ImageViewer } from '../components/viewer/ImageViewer';
import { useRun } from '../hooks/useRun';
import { runsApi } from '../api/runs';
import type { SummaryStats, FrameMetrics, EscalationEvent } from '../types/api';

export const RunResults = () => {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const { status, isProcessing, isPolling, fetchStatus, cancelRun } = useRun();

  const [summary, setSummary] = useState<SummaryStats | null>(null);
  const [metrics, setMetrics] = useState<FrameMetrics[]>([]);
  const [events, setEvents] = useState<EscalationEvent[]>([]);
  const [annotatedFrames, setAnnotatedFrames] = useState<string[]>([]);
  const [eventFrameNames, setEventFrameNames] = useState<Set<string>>(new Set());
  const [viewerOpen, setViewerOpen] = useState(false);
  const [viewerIndex, setViewerIndex] = useState(0);

  // Load data
  useEffect(() => {
    if (!runId) return;
    fetchStatus(runId);
  }, [runId, fetchStatus]);

  // Poll for updates while processing
  useEffect(() => {
    if (!runId || !isProcessing) return;

    const interval = setInterval(async () => {
      await fetchStatus(runId);
    }, 2000);

    return () => clearInterval(interval);
  }, [runId, isProcessing, fetchStatus]);

  // Load results when completed
  useEffect(() => {
    if (!runId || !status?.state) return;
    if (status.state !== 'completed') return;

    const loadResults = async () => {
      try {
        const [s, m, e, a] = await Promise.all([
          runsApi.getSummary(runId),
          runsApi.getMetrics(runId),
          runsApi.getEvents(runId),
          runsApi.listAnnotated(runId),
        ]);
        setSummary(s);
        setMetrics(m);
        setEvents(e.events);
        setAnnotatedFrames(a);
        setEventFrameNames(new Set(e.events.map((ev) => ev.frame_name)));
      } catch (err) {
        console.error('Failed to load results:', err);
      }
    };

    loadResults();
  }, [runId, status?.state]);

  const handleCancel = async () => {
    if (!runId) return;
    await cancelRun(runId);
  };

  const openViewer = (index: number) => {
    setViewerIndex(index);
    setViewerOpen(true);
  };

  if (!runId) return null;

  const isComplete = status?.state === 'completed';

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate('/')}
                className="p-2 hover:bg-slate-100 rounded"
              >
                <ArrowLeft className="h-5 w-5" />
              </button>
              <div>
                <h1 className="text-xl font-bold">Run {runId}</h1>
                <p className="text-sm text-slate-500">
                  {isComplete ? 'Processing complete' : isProcessing ? 'Processing...' : status?.state}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {!isComplete && (
                <Button variant="secondary" onClick={() => fetchStatus(runId)}>
                  <RefreshCw className={`h-4 w-4 mr-2 ${isPolling ? 'animate-spin' : ''}`} />
                  Refresh
                </Button>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {/* Progress */}
          {status && (
            <ProgressMonitor status={status} onCancel={handleCancel} />
          )}

          {/* Results */}
          {isComplete && (
            <>
              <SummaryCards summary={summary} />

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2">
                  <MetricsTable metrics={metrics} eventFrameNames={eventFrameNames} />
                </div>
                <div>
                  <ClassificationChart summary={summary} />
                </div>
              </div>

              <EventsTimeline events={events} runId={runId} />

              <PlotsDisplay runId={runId} />

              <Downloads runId={runId} />

              {/* Frame Gallery */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Annotated Frames</CardTitle>
                    <span className="text-sm text-slate-500">{annotatedFrames.length} frames</span>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2">
                    {annotatedFrames.slice(0, 24).map((frame, idx) => (
                      <button
                        key={frame}
                        onClick={() => openViewer(idx)}
                        className="relative aspect-video bg-slate-100 rounded overflow-hidden hover:ring-2 ring-primary-500"
                      >
                        <img
                          src={`/api/runs/${runId}/artifacts/annotated/${frame}`}
                          alt={frame}
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                        {eventFrameNames.has(frame) && (
                          <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
                        )}
                      </button>
                    ))}
                    {annotatedFrames.length > 24 && (
                      <button
                        onClick={() => openViewer(24)}
                        className="aspect-video bg-slate-100 rounded flex items-center justify-center text-slate-500"
                      >
                        +{annotatedFrames.length - 24} more
                      </button>
                    )}
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </main>

      {/* Image Viewer */}
      {viewerOpen && (
        <ImageViewer
          runId={runId}
          frameNames={annotatedFrames}
          currentIndex={viewerIndex}
          mode="annotated"
          isEvent={eventFrameNames.has(annotatedFrames[viewerIndex])}
          onClose={() => setViewerOpen(false)}
          onNavigate={setViewerIndex}
        />
      )}
    </div>
  );
};
