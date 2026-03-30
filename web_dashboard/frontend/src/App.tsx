/**
 * Main App with routing
 */
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Dashboard } from './pages/Dashboard';
import { RunCreate } from './pages/RunCreate';
import { RunResults } from './pages/RunResults';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      retry: 2,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/new" element={<RunCreate />} />
          <Route path="/runs/:runId" element={<RunResults />} />
        </Routes>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
