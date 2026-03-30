/**
 * Plots display component
 */
import { Download } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { Button } from '../ui/Button';
import { API_BASE_URL } from '../../api/runs';

interface PlotsDisplayProps {
  runId: string;
}

export const PlotsDisplay = ({ runId }: PlotsDisplayProps) => {

  const densityUrl = `${API_BASE_URL}/api/runs/${runId}/artifacts/density_plot.png`;
  const agitationUrl = `${API_BASE_URL}/api/runs/${runId}/artifacts/agitation_plot.png`;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Density Plot */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Crowd Density Trend</CardTitle>
            <a
              href={densityUrl}
              download
              className="text-primary-600 hover:text-primary-700"
            >
              <Button variant="ghost" size="sm">
                <Download className="h-4 w-4" />
              </Button>
            </a>
          </div>
        </CardHeader>
        <CardContent>
          <img
            src={densityUrl}
            alt="Crowd Density Trend"
            className="w-full h-auto rounded"
            onError={() => {}}
          />
        </CardContent>
      </Card>

      {/* Agitation Plot */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Agitation Index Trend</CardTitle>
            <a
              href={agitationUrl}
              download
              className="text-primary-600 hover:text-primary-700"
            >
              <Button variant="ghost" size="sm">
                <Download className="h-4 w-4" />
              </Button>
            </a>
          </div>
        </CardHeader>
        <CardContent>
          <img
            src={agitationUrl}
            alt="Agitation Index Trend"
            className="w-full h-auto rounded"
            onError={() => {}}
          />
        </CardContent>
      </Card>
    </div>
  );
};
