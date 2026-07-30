/**
 * Metrics table component
 */
import { useState, useMemo } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { Input } from '../ui/Input';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import type { FrameMetrics } from '../../types/api';

interface MetricsTableProps {
  metrics: FrameMetrics[];
  eventFrameNames?: Set<string>;
}

export const MetricsTable = ({ metrics, eventFrameNames }: MetricsTableProps) => {
  const [search, setSearch] = useState('');
  const [classificationFilter, setClassificationFilter] = useState<string>('all');
  const [minPeople, setMinPeople] = useState<number | ''>('');
  const [maxPeople, setMaxPeople] = useState<number | ''>('');
  const [onlyEvents, setOnlyEvents] = useState(false);
  const [page, setPage] = useState(1);
  const pageSize = 20;

  // Filter metrics
  const filtered = useMemo(() => {
    return metrics.filter((m) => {
      if (search && !m.frame_name.toLowerCase().includes(search.toLowerCase())) return false;
      if (classificationFilter !== 'all' && m.classification !== classificationFilter) return false;
      if (minPeople !== '' && m.people_count < minPeople) return false;
      if (maxPeople !== '' && m.people_count > maxPeople) return false;
      if (onlyEvents && !eventFrameNames?.has(m.frame_name)) return false;
      return true;
    });
  }, [metrics, search, classificationFilter, minPeople, maxPeople, onlyEvents, eventFrameNames]);

  // head_count exists only for CrowdHuman-fine-tuned YOLO runs
  const hasHeadCounts = useMemo(
    () => metrics.some((m) => m.head_count != null),
    [metrics]
  );

  // Paginate (min 1 page so an empty filter result can't drive page to 0)
  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const paginated = filtered.slice((safePage - 1) * pageSize, safePage * pageSize);

  const getClassificationColor = (classification: string) => {
    switch (classification) {
      case 'Low Crowd': return 'success';
      case 'Moderate Crowd': return 'warning';
      case 'High Crowd': return 'error';
      default: return 'default';
    }
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <CardTitle>Frame Metrics ({filtered.length})</CardTitle>
          <div className="flex flex-wrap gap-2">
            <Input
              placeholder="Search frames..."
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(1); }}
              className="w-48"
            />
            <select
              value={classificationFilter}
              onChange={(e) => { setClassificationFilter(e.target.value); setPage(1); }}
              className="rounded-md border-slate-300 text-sm"
            >
              <option value="all">All Classifications</option>
              <option value="Low Crowd">Low Crowd</option>
              <option value="Moderate Crowd">Moderate Crowd</option>
              <option value="High Crowd">High Crowd</option>
            </select>
            <input
              type="number"
              placeholder="Min people"
              value={minPeople}
              onChange={(e) => { setMinPeople(e.target.value === '' ? '' : parseInt(e.target.value)); setPage(1); }}
              className="w-24 rounded-md border-slate-300 text-sm"
            />
            <input
              type="number"
              placeholder="Max people"
              value={maxPeople}
              onChange={(e) => { setMaxPeople(e.target.value === '' ? '' : parseInt(e.target.value)); setPage(1); }}
              className="w-24 rounded-md border-slate-300 text-sm"
            />
            {eventFrameNames && eventFrameNames.size > 0 && (
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={onlyEvents}
                  onChange={(e) => { setOnlyEvents(e.target.checked); setPage(1); }}
                />
                Events only
              </label>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent noPadding>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Frame</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">People</th>
                {hasHeadCounts && (
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Heads</th>
                )}
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Density</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Agitation</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Classification</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Det (ms)</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Pose (ms)</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-slate-200">
              {paginated.map((m) => (
                <tr key={m.frame_name} className="hover:bg-slate-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">
                    {m.frame_name}
                    {eventFrameNames?.has(m.frame_name) && (
                      <span className="ml-2 text-red-500">*</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">{m.people_count}</td>
                  {hasHeadCounts && (
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">
                      {m.head_count ?? '—'}
                    </td>
                  )}
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">
                    {m.density_ratio.toFixed(3)}×
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">
                    {m.agitation_index.toFixed(4)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <Badge variant={getClassificationColor(m.classification)}>
                      {m.classification}
                    </Badge>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">
                    {(m.inference_time_det * 1000).toFixed(0)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600">
                    {(m.inference_time_pose * 1000).toFixed(0)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="flex items-center justify-between px-6 py-4 border-t">
          <span className="text-sm text-slate-500">
            Page {safePage} of {totalPages}
          </span>
          <div className="flex gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setPage(Math.max(1, safePage - 1))}
              disabled={safePage <= 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setPage(Math.min(totalPages, safePage + 1))}
              disabled={safePage >= totalPages}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
