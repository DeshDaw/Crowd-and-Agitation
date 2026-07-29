/**
 * Downloads component
 */
import { Download, FileJson, Image, Database } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';

interface DownloadsProps {
  runId: string;
}

export const Downloads = ({ runId }: DownloadsProps) => {
  const downloads = [
    {
      label: 'Summary JSON',
      url: `/api/runs/${runId}/artifacts/summary.json`,
      icon: FileJson,
      description: 'Aggregate statistics',
    },
    {
      label: 'Metrics JSON',
      url: `/api/runs/${runId}/artifacts/metrics.json`,
      icon: FileJson,
      description: 'Per-frame metrics',
    },
    {
      label: 'Events JSON',
      url: `/api/runs/${runId}/artifacts/events.json`,
      icon: FileJson,
      description: 'Escalation events timeline',
    },
    {
      label: 'Density Plot',
      url: `/api/runs/${runId}/artifacts/density_plot.png`,
      icon: Image,
      description: 'Crowd density trend',
    },
    {
      label: 'Agitation Plot',
      url: `/api/runs/${runId}/artifacts/agitation_plot.png`,
      icon: Image,
      description: 'Agitation index trend',
    },
    {
      label: 'Database',
      url: `/api/runs/${runId}/artifacts/database.db`,
      icon: Database,
      description: 'SQLite database',
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Download Results</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {downloads.map((item) => (
            <a
              key={item.label}
              href={item.url}
              download
              className="flex items-center gap-4 p-4 border rounded-lg hover:border-primary-500 hover:bg-slate-50 transition-colors"
            >
              <item.icon className="h-8 w-8 text-primary-600" />
              <div className="flex-1">
                <p className="font-medium text-slate-900">{item.label}</p>
                <p className="text-sm text-slate-500">{item.description}</p>
              </div>
              <Download className="h-5 w-5 text-slate-400" />
            </a>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
