import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import type { KillSwitchResponse } from '../../api/types';

type Step = 'idle' | 'confirm' | 'input';

export default function KillSwitch() {
  const [step, setStep] = useState<Step>('idle');
  const [inputValue, setInputValue] = useState('');

  const killMutation = useMutation({
    mutationFn: async () => {
      const res = await apiClient.post<KillSwitchResponse>('/kill-switch', {
        confirm: true,
      });
      return res.data;
    },
    onSuccess: (data) => {
      alert(
        `Kill Switch Activated!\nStrategies stopped: ${data.strategies_stopped}`
      );
      resetState();
    },
    onError: () => {
      alert('Kill Switch failed. Check server logs.');
    },
  });

  const resetState = () => {
    setStep('idle');
    setInputValue('');
  };

  const canExecute = inputValue.toUpperCase() === 'STOP';

  return (
    <div className="bg-red-50 border-2 border-red-300 rounded-lg p-6">
      <h3 className="text-lg font-bold text-red-800">Emergency Kill Switch</h3>
      <p className="mt-2 text-sm text-red-600">
        Stop all strategies immediately and cancel open orders.
      </p>

      {step === 'idle' && (
        <button
          onClick={() => setStep('confirm')}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded font-bold hover:bg-red-700"
        >
          EMERGENCY STOP
        </button>
      )}

      {step === 'confirm' && (
        <div className="mt-4 space-y-3">
          <p className="text-red-700 font-medium">Are you sure you want to proceed?</p>
          <div className="flex gap-2">
            <button
              onClick={() => setStep('input')}
              className="px-4 py-2 bg-red-700 text-white rounded hover:bg-red-800"
            >
              Yes, continue
            </button>
            <button
              onClick={resetState}
              className="px-4 py-2 bg-gray-300 rounded hover:bg-gray-400"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {step === 'input' && (
        <div className="mt-4 space-y-3">
          <p className="text-red-700">Type "STOP" to confirm:</p>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            className="border-2 border-red-300 rounded px-3 py-2 w-32 uppercase"
            placeholder="STOP"
            autoFocus
          />
          <div className="flex gap-2">
            <button
              onClick={() => killMutation.mutate()}
              disabled={!canExecute || killMutation.isPending}
              className={`px-4 py-2 rounded font-bold ${
                canExecute
                  ? 'bg-red-700 text-white hover:bg-red-800'
                  : 'bg-gray-300 cursor-not-allowed'
              }`}
            >
              {killMutation.isPending ? 'Processing...' : 'EXECUTE KILL'}
            </button>
            <button
              onClick={resetState}
              className="px-4 py-2 bg-gray-300 rounded hover:bg-gray-400"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
