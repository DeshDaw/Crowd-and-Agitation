/**
 * Plots display component
 */
import { useState } from 'react';
import { Download, ImageOff } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { Button } from '../ui/Button';

interface PlotsDisplayProps {
  runId: string;
}

const PlotCard = ({ title, url, alt }: { title: string; url: string; alt: string }) => {
  const [failed, setFailed] = useState(false);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>{title}</CardTitle>
          {!failed && (
            <a href={url} download className="text-primary-600 hover:text-primary-700">
              <Button variant="ghost" size="sm">
                <Download className="h-4 w-4" />
              </Button>
            </a>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {failed ? (
          <div className="flex flex-col items-center gap-2 py-10 text-slate-400">
            <ImageOff className="h-8 w-8" />
            <span className="text-sm">
              Plot not available (plot generation may have been disabled)
            </span>
          </div>
        ) : (
          <img
            src={url}
            alt={alt}
            className="w-full h-auto rounded"
            onError={() => setFailed(true)}
          />
        )}
      </CardContent>
    </Card>
  );
};

export const PlotsDisplay = ({ runId }: PlotsDisplayProps) => {
  // Relative URLs: same-origin via the vite dev proxy (and a reverse proxy in
  // production), which also lets the browser honor the download attribute
  const densityUrl = `/api/runs/${runId}/artifacts/density_plot.png`;
  const agitationUrl = `/api/runs/${runId}/artifacts/agitation_plot.png`;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <PlotCard title="Crowd Density Trend" url={densityUrl} alt="Crowd Density Trend" />
      <PlotCard title="Agitation Index Trend" url={agitationUrl} alt="Agitation Index Trend" />
    </div>
  );
};
