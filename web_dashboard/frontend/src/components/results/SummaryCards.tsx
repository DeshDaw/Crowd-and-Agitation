/**
 * Summary statistics cards
 */
import { Users, Gauge, AlertTriangle, Activity, Ruler } from 'lucide-react';
import { Card } from '../ui/Card';
import type { SummaryStats } from '../../types/api';

interface SummaryCardsProps {
  summary: SummaryStats | null;
}

export const SummaryCards = ({ summary }: SummaryCardsProps) => {
  if (!summary) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <Card key={i} className="h-24 animate-pulse bg-slate-100" noPadding><span /></Card>
        ))}
      </div>
    );
  }

  // Optional-chain every numeric field: a degenerate summary.json (empty
  // batch) must render placeholders, not throw and unmount the page.
  // density_ratio is a bbox-area/image-area ratio that legitimately exceeds
  // 1.0 (overlapping boxes), so it renders as a ratio, not a percentage.
  const events = summary.total_escalation_events ?? 0;
  const cards = [
    {
      icon: Users,
      label: 'Avg Crowd Size',
      value: summary.average_crowd_count?.toFixed(1) ?? '—',
      subtext: `±${summary.std_crowd_count?.toFixed(1) ?? '0.0'}`,
      color: 'text-blue-600',
    },
    {
      icon: Gauge,
      label: 'Peak Density',
      value: `${summary.peak_density_value?.toFixed(2) ?? '—'}×`,
      subtext: summary.peak_density_frame || 'bbox area / image area',
      color: 'text-purple-600',
    },
    {
      icon: Activity,
      label: 'Avg Agitation',
      value: summary.mean_agitation?.toFixed(3) ?? '—',
      subtext: `Peak: ${summary.highest_agitation_value?.toFixed(3) ?? '—'}`,
      color: 'text-orange-600',
    },
    {
      icon: AlertTriangle,
      label: 'Escalation Events',
      value: events.toString(),
      subtext: events === 1 ? 'event detected' : 'events detected',
      color: events > 0 ? 'text-red-600' : 'text-green-600',
    },
  ];

  // Calibrated runs get a metric-density card (Fruin Level of Service)
  if (summary.mean_persons_per_m2 != null) {
    const worst = summary.worst_los ?? '—';
    cards.push({
      icon: Ruler,
      label: 'Ground Density',
      value: `${summary.mean_persons_per_m2.toFixed(2)} p/m²`,
      subtext: `worst LOS: ${worst}${summary.mean_speed != null ? ` · ${summary.mean_speed.toFixed(2)} ${summary.speed_unit ?? ''}` : ''}`,
      color: worst <= 'C' ? 'text-green-600' : worst <= 'D' ? 'text-orange-600' : 'text-red-600',
    });
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {cards.map((card) => (
        <Card key={card.label}>
          <div className="flex items-start justify-between">
            <div>
              <p className="text-sm text-slate-500">{card.label}</p>
              <p className="text-2xl font-bold text-slate-900">{card.value}</p>
              <p className="text-xs text-slate-400 mt-1 truncate">{card.subtext}</p>
            </div>
            <card.icon className={`h-6 w-6 ${card.color}`} />
          </div>
        </Card>
      ))}
    </div>
  );
};
