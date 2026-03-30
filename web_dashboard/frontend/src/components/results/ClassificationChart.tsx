/**
 * Crowd classification distribution display
 */
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import type { SummaryStats } from '../../types/api';

interface ClassificationChartProps {
  summary: SummaryStats | null;
}

export const ClassificationChart = ({ summary }: ClassificationChartProps) => {
  if (!summary?.crowd_classification_distribution) {
    return null;
  }

  const distribution = summary.crowd_classification_distribution;
  const total = Object.values(distribution).reduce((a, b) => a + b, 0);

  const items = [
    {
      label: 'Low Crowd',
      count: distribution['Low Crowd'] || 0,
      color: 'bg-green-500',
    },
    {
      label: 'Moderate Crowd',
      count: distribution['Moderate Crowd'] || 0,
      color: 'bg-yellow-500',
    },
    {
      label: 'High Crowd',
      count: distribution['High Crowd'] || 0,
      color: 'bg-red-500',
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Crowd Classification</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {items.map((item) => {
            const percentage = total > 0 ? (item.count / total) * 100 : 0;
            return (
              <div key={item.label}>
                <div className="flex justify-between text-sm mb-1">
                  <span>{item.label}</span>
                  <span className="font-medium">{item.count} ({percentage.toFixed(1)}%)</span>
                </div>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${item.color} transition-all duration-500`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
};
