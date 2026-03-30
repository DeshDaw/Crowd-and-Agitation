/**
 * Main dashboard page with runs list
 */
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Activity, Trash2, AlertTriangle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { StatusBadge } from '../components/ui/StatusBadge';
import { useRunsStore } from '../store/runsStore';
import { runsApi } from '../api/runs';

export const Dashboard = () => {
  const navigate = useNavigate();
  const { runs, refreshRuns } = useRunsStore();

  useEffect(() => {
    refreshRuns();
    const interval = setInterval(refreshRuns, 5000);
    return () => clearInterval(interval);
  }, [refreshRuns]);

  const handleDelete = async (e: React.MouseEvent, runId: string) => {
    e.stopPropagation();
    if (confirm('Delete this run? This cannot be undone.')) {
      await runsApi.delete(runId);
      refreshRuns();
    }
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity className="h-8 w-8 text-primary-600" />
              <div>
                <h1 className="text-xl font-bold text-slate-900">
                  Crowd Surveillance Dashboard
                </h1>
                <p className="text-sm text-slate-500">
                  Abnormal Crowd Motion Detection
                </p>
              </div>
            </div>
            <Button onClick={() => navigate('/new')} size="lg">
              <Plus className="h-5 w-5 mr-2" />
              New Run
            </Button>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Card>
          <CardHeader>
            <CardTitle>Processing Runs ({runs.length})</CardTitle>
          </CardHeader>
          <CardContent noPadding>
            {runs.length === 0 ? (
              <div className="text-center py-12">
                <Activity className="h-12 w-12 text-slate-300 mx-auto mb-4" />
                <p className="text-slate-500 mb-4">No runs yet. Create your first processing run.</p>
                <Button onClick={() => navigate('/new')}>
                  <Plus className="h-4 w-4 mr-2" />
                  Create Run
                </Button>
              </div>
            ) : (
              <div className="divide-y divide-slate-200">
                {runs.map((run) => (
                  <div
                    key={run.run_id}
                    onClick={() => navigate(`/runs/${run.run_id}`)}
                    className="p-4 hover:bg-slate-50 cursor-pointer flex items-center justify-between"
                  >
                    <div className="flex items-center gap-4">
                      <div>
                        <p className="font-medium text-slate-900">Run {run.run_id}</p>
                        <p className="text-sm text-slate-500">
                          {run.total_frames} frames • Created {new Date(run.created_at).toLocaleString()}
                        </p>
                      </div>
                      {run.has_events && (
                        <AlertTriangle className="h-5 w-5 text-red-500" />
                      )}
                    </div>
                    <div className="flex items-center gap-4">
                      <StatusBadge state={run.state} />
                      <button
                        onClick={(e) => handleDelete(e, run.run_id)}
                        className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
};
