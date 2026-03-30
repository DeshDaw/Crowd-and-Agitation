/**
 * Events timeline component
 */
import { AlertTriangle, ExternalLink } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { Button } from '../ui/Button';
import type { EscalationEvent } from '../../types/api';

interface EventsTimelineProps {
  events: EscalationEvent[];
  runId: string;
}

export const EventsTimeline = ({ events, runId }: EventsTimelineProps) => {
  if (events.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Escalation Events</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-slate-500">
            <p>No escalation events detected.</p>
            <p className="text-sm mt-2">
              Events occur when High Crowd density and elevated agitation are detected.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-red-500" />
            <CardTitle>Escalation Events ({events.length})</CardTitle>
          </div>
        </div>
      </CardHeader>
      <CardContent noPadding>
        <div className="divide-y divide-slate-200">
          {events.map((event) => (
            <div key={event.frame_name} className="p-4 hover:bg-slate-50">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-slate-900">{event.frame_name}</p>
                  <p className="text-sm text-slate-500">
                    Frame {event.frame_index + 1} • {event.classification}
                  </p>
                  <div className="mt-2 flex gap-4 text-sm">
                    <span className="text-orange-600">
                      Agitation: {event.agitation_score.toFixed(4)}
                    </span>
                    <span className="text-purple-600">
                      Density: {(event.density_ratio * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
                <a
                  href={`/api/runs/${runId}/artifacts/escalation/${event.frame_name}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Button variant="ghost" size="sm">
                    <ExternalLink className="h-4 w-4" />
                  </Button>
                </a>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
