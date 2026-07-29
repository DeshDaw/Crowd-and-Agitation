/**
 * Main App with routing
 */
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Dashboard } from './pages/Dashboard';
import { RunCreate } from './pages/RunCreate';
import { RunResults } from './pages/RunResults';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import './index.css';

const NotFound = () => (
  <div className="min-h-screen bg-slate-50 flex items-center justify-center">
    <div className="text-center space-y-3">
      <h1 className="text-2xl font-bold text-slate-900">Page not found</h1>
      <Link to="/" className="text-primary-600 text-sm underline">
        Back to Dashboard
      </Link>
    </div>
  </div>
);

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/new" element={<RunCreate />} />
          <Route path="/runs/:runId" element={<RunResults />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
