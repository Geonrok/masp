import type { FallbackProps } from 'react-error-boundary';

export default function ErrorFallback({ error, resetErrorBoundary }: FallbackProps) {
  const showDetail = import.meta.env.DEV;

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-red-50 p-4">
      <div className="bg-white rounded-lg shadow-lg p-8 max-w-md text-center">
        <h2 className="text-2xl font-bold text-red-600 mb-4">Something went wrong</h2>
        <pre className="text-sm text-gray-600 bg-gray-100 p-4 rounded mb-4 overflow-auto max-h-40">
          {showDetail ? error.message : 'Unexpected error. Please retry.'}
        </pre>
        <button
          onClick={resetErrorBoundary}
          className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 transition"
        >
          Try Again
        </button>
      </div>
    </div>
  );
}
